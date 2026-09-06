# UWBRadarLocalization — TinyML Multi-Person 2D Localization from UWB Radar CIR

### Embedded and Edge Artificial Intelligence — Politecnico di Milano (A.Y. 2025-2026)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Target: ESP32-S3](https://img.shields.io/badge/Target-ESP32--S3-blue.svg)](https://www.espressif.com/en/products/socs/esp32-s3)
[![Framework: TFLite Micro](https://img.shields.io/badge/Framework-TFLite%20Micro-orange.svg)](https://www.tensorflow.org/lite/microcontrollers)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)

---

## 📌 Project Overview

**UWBRadarLocalization** is an end-to-end TinyML occupancy-grid system for real-time 2D multi-person localization (up to 4 people) in an indoor environment (4.8 m × 7.2 m) using raw Ultra-Wideband (UWB) radar Channel Impulse Response (CIR) signals.

The sensing infrastructure comprises **6 distributed UWB radar nodes (TSRR250)** mounted at 1.2 m height along the perimeter of the room. Each sensor features an L-shaped 3-antenna array with 7.5 mm spacing, capturing 120 range bins sampled at 25 Hz.

The core neural network and inference pipeline are engineered for resource-constrained edge execution on an **ESP32-S3 microcontroller** using **TensorFlow Lite for Microcontrollers (TFLM)**. The deployed system achieves high localization fidelity while strictly adhering to rigorous micro-architectural hardware constraints:

* **Strict Edge Budget Compliance**: Model file size of **349.5 KB** (< 800 KB flash budget, 44% utilization), activation arena of **26–57 KB** (< 300 KB SRAM budget), 11 operators, zero forbidden recurrent/custom operators, and full **INT8 quantization**.
* **Conditional Inference (Energy Gating)**: A lightweight input-energy threshold (costing only 846 additions vs. ~330,000 MACs for a full network pass) skips **95.1% of empty-room frames** with **0.0000% false skips** on separable occupied frames, reducing edge compute energy by ~99.7% during inactive periods.
* **Temporal Heatmap Persistence**: A causal exponential smoothing filter ($max(P, S \cdot \theta / t_{lo})$) on the output occupancy heatmap preserves detections of stationary/seated subjects who otherwise fade below detection thresholds, lifting seated recall to **68.6%** without retraining.
* **Leave-Window-Out Generalization**: Validated using 6-fold stratified leave-window-out cross-validation across 24 acquisition windows (180,000 frames total). Achieves a micro-averaged **$F_1$ score of 0.8682 ± 0.018** (Hungarian matching @ 1.0 m acceptance radius) on the deployed INT8 quantization path.

---

## 📂 Repository Structure

```text
.
├── evaluation/                      # Evaluation harness and hardware constraint verification
│   ├── evaluate_constraint.py       # Checks flash size, activation arena, and TFLM operator blocklist
│   ├── evaluate_performance.py      # Computes Hungarian matching F1, precision, recall, and metrics
│   └── example/                     # Evaluation test slice (input_test.npy, output_test.jsonl)
├── report/                          # Technical documentation and experimental record
│   ├── report.pdf                   # Compiled research paper / final technical report
│   ├── report.typ                   # Typst source document for the report
│   ├── REPORT_NOTES.md              # Detailed lab notebook & experimental log (all phases & sweeps)
│   ├── images/                      # Coverage plots, CIR signals, and prediction visualizations
│   ├── postproc_sweep.csv           # Post-processing grid search results (1,008 configurations)
│   └── surface_sweep.csv            # Threshold and persistence sweep metrics
├── src/                             # Training, quantization, and experimental pipeline
│   ├── model.py                     # Occupancy-grid architecture (early Dense fusion + SeparableConv2D)
│   ├── preprocessing.py             # Decluttering EMA, range cropping [5:52], magnitude, normalization
│   ├── quantize.py                  # Full INT8 post-training quantization with decluttered calibration
│   ├── tracker.py                   # Centroid tracker with velocity smoothing and coasting
│   ├── train.py                     # Stratified 6-fold leave-window-out training loop
│   ├── channels.py                  # Multi-channel feature extractors (magnitude, phase, respiration)
│   ├── cache.py                     # Disk-cached feature precomputations
│   ├── metrics.py                   # Custom Hungarian matching and evaluation metrics
│   ├── sweep_postproc.py            # Automated parameter search over decoding thresholds and tracking
│   └── helpers/                     # Diagnostic, verification, and visualization utilities
│       ├── calibrate_energy_gate.py # Energy gate calibration for conditional inference
│       ├── verify_parity.py         # Bit-exact parity verification between training and inference paths
│       ├── verify_breath_channel.py # Respiration bandpass filter investigation
│       ├── verify_perio_channel.py  # Periodicity and coherence feature diagnostics
│       ├── visualize_fov.py         # Radar placement and room coverage visualizer
│       └── animate_predictions.py   # Prediction vs ground truth animated visualizer
├── submission/                      # ESP32-S3 deployment deliverables
│   ├── code.py                      # Production inference script (streaming declutter, gating, INT8 TFLite)
│   └── model.tflite                 # Deployed INT8 quantized TFLite model (349.5 KB)
├── project_spec.md                  # Project specification and hardware constraints
├── requirements.txt                 # Runtime dependencies for edge inference
└── requirements-dev.txt             # Development and training dependencies
```

---

## 🛠️ System Architecture

### 1. Signal Processing & Preprocessing
* **Decluttering High-Pass EMA**: Raw CIR signals are dominated by static multipath reflections from walls and furniture. We apply a first-order recursive high-pass filter via an Exponential Moving Average ($y_t = x_t - \text{EMA}_\alpha(x)_t$ with $\alpha = 0.99$), corresponding to a high-pass corner at ~0.04 Hz.
  * Filter state is seeded using `scipy.signal.lfilter_zi` scaled by frame 0 to prevent cold-start transient spikes.
  * Streaming inference in `submission/code.py` uses state-preserving single-frame `lfilter` to maintain bit-exact parity with the batch training pipeline.
* **Range-Bin Crop (`[5:52]`)**: Bins 0–4 contain internal TX→RX antenna crosstalk; bin 52 corresponds to ~7.8 m (covering the 4.8 m × 7.2 m room boundaries). Cropping to 47 bins removes 61% of raw input volume while eliminating out-of-room multipath.
* **Magnitude & Fold-Safe Normalization**: Signal magnitude is extracted via $\sqrt{I^2 + Q^2}$ (avoiding subtle hypot discrepancies) and normalized to $[0, 1]$ via the 99th percentile computed strictly on training folds.

### 2. Neural Network Topology
* **Early Global Fusion**: The 6 distributed radar locations do not share spatial translation invariance. The flattened $(6, 3, 47, 1)$ input is projected through a global `Dense(256)` layer followed by `Dense(432)`, reshaped into an $18 \times 12 \times 2$ spatial feature map.
* **Depthwise-Separable Convolutions**: 3 consecutive depthwise-separable convolution blocks (`SeparableConv2D`) refine spatial features across the room grid.
* **Head & Target Representation**: A $1 \times 1$ linear convolution generates an $18 \times 12$ occupancy heatmap corresponding to 0.4 m × 0.4 m grid cells. Targets are generated as 2D Gaussian blobs ($\sigma = 0.3$ m) centered at each person's position.
* **Sub-Cell Peak Decoding**: Heatmap peaks are extracted via local maximum detection, refined to sub-cell precision through a $3 \times 3$ center-of-mass calculation, and filtered via Non-Maximum Suppression (NMS).

### 3. Edge Optimization & INT8 Quantization
* **BatchNorm Operator Folding**: Reordered layers to `Linear → BatchNorm → ReLU`. This enables the TFLite converter to fold batch normalization parameters directly into the preceding convolution/dense weights, collapsing operator count from **20 to 11**.
* **Constant Reshape Folding**: Exporting with a fixed batch-1 input signature eliminates dynamic shape-slicing operations (`SHAPE → STRIDED_SLICE → PACK → RESHAPE`), leaving a single static `RESHAPE`.
* **Representative Calibration**: Quantized to full `int8` using decluttered calibration frames matching the actual runtime deployment distribution, avoiding calibration range blow-up.

### 4. Conditional Inference & Output Persistence
* **Conditional Gating**: On-device host code computes the mean normalized magnitude across all 846 input cells. If below the calibrated energy gate, inference is bypassed completely, emitting an empty heatmap.
* **Heatmap Persistence**: A causal exponential moving average filter ($\tau = 10\text{ s}$) maintains temporal context on output heatmaps. Stationary individuals who produce weak instantaneous doppler signals are decoded via $max(P, S \cdot \theta / t_{lo})$, preventing false negatives for seated persons.

---

## 📊 Performance & Edge Benchmarks

### 1. ESP32-S3 Hardware Constraints

All hard hardware checks enforced by `evaluation/evaluate_constraint.py` pass with wide safety margins:

| Hardware Constraint | Target Limit | Measured Value | Margin | Status |
|---|---|---|---|:---:|
| **Flash Budget (File Size)** | < 800 KB | **349.5 KB** | 2.3× safety margin (44% budget) | ✅ Pass |
| **Activation Arena (SRAM)** | < 300 KB | **26.2 – 57.4 KB** | 5.2× – 11.4× safety margin | ✅ Pass |
| **Operator Count** | — | **11 operators** (down from 20) | Fully folded BatchNorms | ✅ Pass |
| **Forbidden Operators** | No LSTM/RNN/CUSTOM | **0 forbidden ops** | Full TFLM compatibility | ✅ Pass |
| **Quantization Scheme** | INT8 recommended | **Full INT8** | Weights & activations quantized | ✅ Pass |

### 2. Localization Performance (Leave-Window-Out CV)

Cross-validation evaluated across all 24 acquisition windows (stratified by subject count) scored with Hungarian matching @ 1.0 m threshold:

| Metric | Score / Value | Evaluation Notes |
|---|---|---|
| **Pooled $F_1$ Score** | **0.8682 ± 0.018** | Evaluated on full INT8 deployment path ($\theta = 0.50$) |
| **Seated Subject Recall** | **68.6%** | Recovered via temporal heatmap persistence |
| **Empty Frame Skip Rate** | **95.1%** | Computed via conditional inference energy gate |
| **Occupied Frame False Skip Rate** | **0.0000%** | Zero false skips on separable occupied windows |
| **Compute Saving (Empty Frame)** | **~99.7%** | 846 additions vs. ~330,000 MACs |

---

## 🚀 Getting Started

### 📋 Prerequisites
* Python 3.10+ (tested with Python 3.10 and 3.11)
* Virtual environment (`venv`)

### ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/M1RK02/UWBRadarLocalization.git
   cd UWBRadarLocalization
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Optional: for training and diagnostics
   ```

### 🧠 Running Edge Inference
Execute the production deployment script on test CIR data:
```bash
python submission/code.py \
    --input-path evaluation/example/input_test.npy \
    --output-path evaluation/example/my_output.jsonl
```

### 📈 Evaluating Performance & Metrics
Evaluate Hungarian matching $F_1$, precision, recall, and localization errors:
```bash
python evaluation/evaluate_performance.py \
    --gt-path evaluation/example/output_test.jsonl \
    --pred-path evaluation/example/my_output.jsonl
```

### 🔬 Validating ESP32-S3 Hardware Constraints
Verify that `submission/model.tflite` meets all flash, SRAM activation arena, and operator constraints:
```bash
python evaluation/evaluate_constraint.py \
    --model-path submission/model.tflite
```

### 🏋️ Reproducing Training & Post-Processing
To retrain the 6-fold leave-window-out model pipeline:
```bash
# Optional: Download dataset from Hugging Face (requires org access + HF token)
python src/helpers/download_dataset.py

# Run 6-fold stratified training and quantization pipeline
python src/train.py

# Monitor training metrics with TensorBoard
tensorboard --logdir logs
```

---

## 🤝 Contributors

* **Mirko Pica** ([@M1RK02](https://github.com/M1RK02)) — Model Architecture, Quantization & Operator Folding, Conditional Inference Gating, Evaluation Harness.
* **Christian Prendin** ([@ChristianPrendin](https://github.com/ChristianPrendin)) — Signal Processing Pipeline, Temporal Heatmap Persistence, Seated Subject Diagnostics, Post-Processing Optimization.

Developed for the *Embedded and Edge Artificial Intelligence* course at **Politecnico di Milano** (Department of Electronics, Information and Bioengineering).

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
