# Embedded and Edge Artificial Intelligence — Project Assignment

**Politecnico di Milano | Department of Electronics, Information and Bioengineering**

---

## 1. Task

Build a **TinyML system** that estimates the **2D positions of up to 4 people** inside a room using raw **UWB radar signals** (Channel Impulse Response — CIR).

| Property             | Value                                          |
| -------------------- | ---------------------------------------------- |
| Input                | `.npy` file — shape `(T, 6, 3, 120, 2)`       |
| Output               | `.jsonl` file — one JSON object per frame      |
| Target hardware      | ESP32-S3 via TensorFlow Lite for Microcontrollers |
| Max people per frame | 4 (minimum 0)                                  |
| Room dimensions      | 4.8 m × 7.2 m                                 |

The final evaluation uses a **separate acquisition session not seen during training**. Build a model that generalises — overfitting the training data will not help.

---

## 2. Teams & Deliverables

**Teams:** groups of exactly 2 students, submitted via GitHub Classroom.
Group name format: `<person_code_1>-<person_code_2>` (e.g. `10xxxxxx-10yyyyyy`).

**GitHub Classroom invitation:** <https://classroom.github.com/a/8dqcBAR3>

**Deliverables:**

| File                 | Location       | Description                           |
| -------------------- | -------------- | ------------------------------------- |
| `code.py`            | `submission/`  | Inference script (exact name)         |
| `model.tflite`       | `submission/`  | Trained TFLite model (exact name)     |
| `report.pdf`         | `report/`      | Project report                        |
| Training code        | `src/`         | Notebooks, scripts, experiments       |

**Submission deadlines:**

- **July 15** — June/July session
- **September 1** — September session

---

## 3. Dataset

### Source

Published on Hugging Face (private organisation):
<https://huggingface.co/datasets/HAEEAI/multi-person-localization>

Join the HAEEAI organisation first:
<https://huggingface.co/organizations/HAEEAI/share/BWZUHibZiaCeOKdfJNlFEXHHwvVFdlUnfR>

### Sensor Setup

- **6 fixed UWB radar sensors** (TSRR250), placed around the room perimeter at **1.2 m height**
- Each radar has **3 antennas** (L-shape array, 7.5 mm spacing)
- **120 range bins** per antenna, sampled at **25 Hz**
- **1 bin ≈ 0.15 m** — total physical range ≈ 18 m
- **Field of view ≈ 120°** (±60° around boresight)
- **First 5 bins** contain direct TX→RX coupling and are typically discarded

**Radar positions** (from `sensor_setup.json`, `[x, y, z]` in metres):

| Radar    | Position         | Facing |
| -------- | ---------------- | ------ |
| SR250_1  | `(2.4, 0.0, 1.2)` | Up     |
| SR250_2  | `(4.8, 1.8, 1.2)` | Left   |
| SR250_3  | `(4.8, 5.4, 1.2)` | Left   |
| SR250_4  | `(2.4, 7.2, 1.2)` | Down   |
| SR250_5  | `(0.0, 5.4, 1.2)` | Right  |
| SR250_6  | `(0.0, 1.8, 1.2)` | Right  |

### Coordinate System

- Origin at the **bottom-left corner** of the room
- **x** axis: positive to the right (0 → 4.8 m)
- **y** axis: positive upward (0 → 7.2 m)
- All positions in **metres**

### Data Format (`.npz` per acquisition window)

| Array          | Shape            | Dtype   | Description                                                     |
| -------------- | ---------------- | ------- | --------------------------------------------------------------- |
| `radar_cir_iq` | `(T, 6, 3, 120, 2)` | float32 | Complex CIR: T frames × 6 radars × 3 antennas × 120 bins × [I, Q] |
| `people_xy`    | `(T, 4, 2)`     | float32 | Ground-truth [x, y] positions (zero-padded)                     |
| `people_mask`  | `(T, 4)`        | uint8   | Boolean mask — True for active persons                          |
| `timestamps`   | `(T,)`          | float32 | Acquisition timestamps                                          |

The complex CIR is reconstructed as:

```python
cir = radar_cir_iq[..., 0] + 1j * radar_cir_iq[..., 1]  # shape (T, 6, 3, 120)
```

### Acquisition Windows

24 windows, each 300 s (7 500 frames) — 180 000 frames total (120 minutes).
Scenarios range from 0 to 4 subjects with varying movement patterns:

| Window   | Description                                            |
| -------- | ------------------------------------------------------ |
| 000000   | Random walk of 2 subjects                              |
| 000001   | Random walk of 2 subjects, split left/right of x=2.4   |
| 000002   | Random walk of 2 subjects, split above/below y=3.6     |
| 000003   | Random walk of 2 subjects                              |
| 000004   | Random walk of 1 subject, 1 seated                     |
| 000005–7 | Random walk of 4 subjects                              |
| 000008–9 | Random walk of 3 subjects, 1 seated                    |
| 000010–11| Random walk of 2 subjects, 2 seated                    |
| 000012–13| Random walk of 3 subjects                              |
| 000014   | Random walk of 2 subjects, 1 seated                    |
| 000015   | Random walk of 1 subject, 2 seated                     |
| 000016–18| Random walk of 1 subject                               |
| 000019   | Serpentine walk of 1 subject (long side)                |
| 000020   | Serpentine walk of 1 subject (short side)               |
| 000021   | 1 subject seated                                       |
| 000022–23| Empty room (0 subjects)                                |

**Radar selection is a design choice.** You may use any subset of the 6 radars. More radars = richer signal but larger model input. Experiment and justify in the report.

---

## 4. Input / Output Specification

### Input to `code.py`

A single `.npy` file of shape `(T, 6, 3, 120, 2)` — raw CIR measurements without ground truth.

### Output from `code.py`

A `.jsonl` file with **one JSON object per frame**:

```jsonl
{"frame": 36, "localizations": [[1.2, 3.4], [2.5, 5.1]]}
{"frame": 37, "localizations": [[1.3, 3.5]]}
{"frame": 38, "localizations": []}
```

| Field           | Type              | Description                                                  |
| --------------- | ----------------- | ------------------------------------------------------------ |
| `frame`         | int               | 0-based frame index                                          |
| `localizations` | list of `[x, y]`  | Predicted positions in metres; empty `[]` if no person detected |

### How `code.py` is called

```bash
python submission/code.py \
    --input-path  <path/to/input.npy> \
    --output-path <path/to/output.jsonl>
```

### Constraints on `code.py`

Only the packages listed in `requirements.txt` may be imported:

| Package      | Version  | Permitted use                                  |
| ------------ | -------- | ---------------------------------------------- |
| `numpy`      | ≥ 2.0.0  | All array operations and preprocessing         |
| `scipy`      | ≥ 1.10.0 | `ndimage`, `signal`, `spatial` submodules      |
| `tensorflow`  | ≥ 2.13.0 | `tf.lite.Interpreter` only — not `tf.keras`    |

Python standard library modules (`argparse`, `json`, `sys`, `pathlib`, `math`, `collections`, `itertools`, …) are always permitted. Any other third-party import causes **evaluation failure**.

File names must be exactly `code.py` and `model.tflite`.

---

## 5. ESP32-S3 Deployment Constraints

Your model will **not** be deployed on real hardware, but it **must meet** these constraints (verified by `evaluate_constraint.py`):

| Check                       | Limit           | Notes                                          |
| --------------------------- | --------------- | ---------------------------------------------- |
| Model file size             | **< 800 KB**    | Flash budget                                   |
| Activation arena (SRAM)     | **< 300 KB**    | Model arena only — preprocessing also uses SRAM |
| Quantization                | INT8 recommended | 4× smaller and faster on ESP32-S3              |
| Forbidden ops               | **No LSTM / GRU / RNN / CUSTOM / Flex** | Not supported by TFLM     |

**Supported layers:** `Conv2D`, `DepthwiseConv2D`, `Dense`, `BatchNormalization`, `MaxPool2D`, `AveragePooling2D`, `Add`, `ReLU`. Full list: `tensorflow/tflite-micro` — `micro_mutable_op_resolver.h`.

**Important:** LSTM/GRU layers *can* be converted to TFLite successfully (the conversion will not fail), but TensorFlow Lite for Microcontrollers does not support them. The constraint checker catches this.

### The 5 Constraint Checks

| #  | Check                  | Type     | Details                                                      |
| -- | ---------------------- | -------- | ------------------------------------------------------------ |
| 1  | TFLite validity        | Hard     | Model must load and allocate tensors                         |
| 2  | File size              | Hard     | `model.tflite` < 800 KB                                     |
| 3  | Activation arena       | Hard     | Estimated via dummy inference; only activation tensors counted (not weights) — < 300 KB |
| 4  | Forbidden operations   | Hard     | LSTM, GRU, RNN variants, CUSTOM ops, Flex ops                |
| 5  | Quantization           | Advisory | Full INT8 → PASS; hybrid → WARNING; float32 → WARNING        |

**Exit codes:** 0 = all hard constraints pass | 1 = at least one failed.

---

## 6. Evaluation Metrics

### F1 Score (primary detection metric)

**Step 1 — Format validation:**
The `.jsonl` file is validated before scoring: valid JSON, no duplicate frames, max 4 localisations per frame, all coordinates within room bounds [0, 4.8] × [0, 7.2] m. Any error → script exits, no score.

**Step 2 — Hungarian matching per frame:**
Predicted positions are matched to ground-truth persons using optimal assignment (Hungarian algorithm) with a **1.0 m acceptance threshold**:

- **TP**: predicted position matched to a GT person within 1.0 m
- **FP**: predicted position with no GT match (or distance > 1.0 m)
- **FN**: GT person not detected

**Step 3 — Global F1 aggregation:**

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

Frames emitted by your code but not present in the ground truth are ignored. Ground-truth frames with no prediction are treated as all False Negatives.

### Additional Metrics (informational only)

Reported on matched (TP) pairs only:

- **RMSE**, **MAE**, **Median error**, **P90**
- **Count MAE**, **Count accuracy** — person count per frame

---

## 7. Repository Structure

```
submission/
    code.py              ← inference script (required, exact name)
    model.tflite         ← trained TFLite model (required, exact name)

report/
    report.pdf           ← project report

src/
    ...                  ← training notebooks, scripts, experiments

evaluation/              ← provided — do not modify
    evaluate_performance.py
    evaluate_constraint.py
    example/
        input_test.npy
        output_test.jsonl

requirements.txt         ← allowed libraries — do not modify
```

The `evaluation/` folder and `requirements.txt` are **protected**. Do not modify their contents.

---

## 8. Local Testing

Run all three steps before every submission.

**Step 1 — Run inference:**

```bash
python submission/code.py \
    --input-path  evaluation/example/input_test.npy \
    --output-path evaluation/example/my_output.jsonl
```

**Step 2 — Check performance:**

```bash
python evaluation/evaluate_performance.py \
    --gt-path   evaluation/example/output_test.jsonl \
    --pred-path evaluation/example/my_output.jsonl
```

**Step 3 — Check ESP32-S3 constraints:**

```bash
python evaluation/evaluate_constraint.py \
    --model-path submission/model.tflite
```

> The example input/output is derived from training data and is provided for **format verification and sanity checking only**. The final evaluation uses a separate acquisition session not seen during training.

---

## 9. Grading

The project is worth **12 points out of 30**:

| Component              | Points |
| ---------------------- | ------ |
| Baseline Accuracy      | 3      |
| Edge Optimization      | 3      |
| Documentation          | 2      |
| Code and Reproducibility | 2    |
| Innovation / Ranking   | 2      |

**Innovation/Ranking** rewards creative approaches and competitive accuracy. Groups are ranked: higher-ranked, innovative solutions earn more points.

The written exam and the project may be in different sessions, as long as both are completed within the same academic year.

Pre-existing models (e.g. from Hugging Face) are allowed, provided they respect all project constraints.

---

## 10. Q&A

**Can we use pre-existing models?**
Yes, but they must respect all project constraints (size, arena, ops, quantization).

**Can groups be of 3 students?**
No. Groups must be exactly 2 students.

**Can the exam and project be in different sessions?**
Yes, within the same academic year.

**What does "Innovation/Ranking" mean?**
It rewards creative approaches and competitive accuracy. Groups are ranked; higher-ranked, innovative solutions earn more points.
