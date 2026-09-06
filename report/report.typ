#set page(paper: "a4", margin: 2.2cm, numbering: "1")
#set text(font: "Libertinus Serif", size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => [#v(0.6em) #it #v(0.2em)]
#show raw: set text(font: "DejaVu Sans Mono", size: 8.5pt)

#let finding(body) = block(
  fill: luma(245), inset: 8pt, radius: 3pt, width: 100%, body,
)

#align(center)[
  #text(17pt, weight: "bold")[Multi-Person Localization from UWB Radar CIR]
  #v(0.3em)
  #text(11pt)[A TinyML occupancy-grid system for the ESP32-S3]
  #v(0.6em)
  #text(10pt)[Mirko Pica · Christian Prendin]
  #v(0.2em)
  #text(9pt, style: "italic")[Embedded and Edge Artificial Intelligence — Politecnico di Milano]
]

#v(1em)

#outline(depth: 2, indent: 1em)

#v(1em)

= Problem and constraints

The system estimates the 2D positions of up to four people in a
4.8 m × 7.2 m room from raw UWB radar channel impulse responses: six radars,
three antennas each, 120 range bins, complex $I/Q$, sampled at 25 Hz. The
trained model must run on an ESP32-S3 through TensorFlow Lite for
Microcontrollers, under a 800 KB flash budget, a 300 KB activation arena, and a
blocklist that rules out every recurrent operator. Scoring is global
micro-averaged $F_1$ with Hungarian matching at a 1.0 m radius.

Final evaluation uses a separate acquisition session from the one used during
development, so generalization across sessions — not just across held-out
frames of the same recordings — is the relevant test of this system. All
validation figures in this report use a leave-window-out cross-validation
estimate rather than a same-session holdout, reported with its uncertainty.

= Signal path

== Decluttering: an EMA as a 0.04 Hz high-pass

A static room dominates the raw CIR. We subtract an exponential moving average
of the signal from itself, $y_t = x_t - "EMA"_alpha (x)_t$ with $alpha = 0.99$,
which is a first-order high-pass with a corner near 0.04 Hz. Everything
stationary cancels; everything that moves survives.

Two details are load-bearing and neither is obvious:

- *The filter state is seeded from frame 0*, not from zero. Initialized at zero,
  the filter spends its first seconds reporting the entire static room as one
  enormous movement. `scipy.signal.lfilter_zi` gives the steady state for a
  constant input; scaling it by the first frame starts the filter as if the
  scene had been frozen there forever.
- *The streaming and batch implementations use the same code path.* Deployment
  applies `lfilter` to one frame at a time, carrying its filter state between
  calls, which reproduces the batch computation bit-for-bit. A more
  direct-looking recursion, `bg = alpha*bg + (1-alpha)*x`, is not equivalent:
  its arithmetic order differs from `lfilter`'s internal implementation, and the
  two diverge by $~4 times 10^(-2)$ over a recording.

One direct consequence of this, which we confirmed rather than assumed: because
the EMA seeds from frame 0, the decluttered frame 0 is identically zero in every
recording. That is a cold-start artifact, and it reappears in Section 6.

== Range-bin crop and the magnitude channel

#figure(
  image("images/fov_visualization.png", width: 88%),
  caption: [Coverage of the six radars over the 4.8 m × 7.2 m room, and the
    mapping from range bins to physical distance. The crop to bins 5–52 removes
    only the sensor's own coupling and the bins beyond the room.],
)

Bins 0–4 carry transmit-to-receive coupling inside the sensor itself; bin 52
corresponds to roughly 7.8 m of range, just short of the room's 8.65 m diagonal,
so the crop keeps essentially the full room while excluding bins that can only
be multipath. Cropping to `[5:52]` removes 61% of the input.

#figure(
  image("images/radar_signal_plot.png", width: 88%),
  caption: [Decluttered magnitude for a single radar and antenna over one
    recording, with two tracked person trajectories overlaid — this is close to
    the representation the network actually receives, after the magnitude and
    normalization steps below.],
)

The model is then fed $|I + j Q|$, normalized by a per-channel 99th percentile
computed *on each fold's training windows only*. One subtlety survives into the
code as a comment: the magnitude is computed as `sqrt(re**2 + im**2)` and never
as `np.abs` on a complex array, because `np.abs` uses a hypot-style algorithm
whose result differs in the last bits. The cached training data and the deployed
model were both built on the `sqrt` form, and mixing them silently breaks
byte-level parity between training and inference.

Putting the chain together: the raw complex CIR is decluttered (§2.1), cropped
to the 47 informative range bins, reduced to magnitude, and clipped and rescaled
to $[0, 1]$ by the per-fold 99th percentile. The deployed model uses this single
magnitude channel per radar and antenna; Section 7 describes seven additional
channel sets that were tried against it and rejected.

= Model architecture

#finding[
  *Why the first layer is a `Dense`, not a convolution.* The input axis of length
  6 indexes *radars*, which are physically distributed around the room. A
  convolution over that axis would assert that radar 1 and radar 2 are
  neighbours in a way radar 1 and radar 6 are not — a translation-invariance
  claim about an axis that has no translation structure at all. The geometric
  mapping from six range-profile sets to one room is arbitrary and global, so it
  is learned by a fully-connected projection.
]

The network flattens the $(6, 3, 47, 1)$ input and projects it through a
256-unit `Dense` layer, then a second, 432-unit `Dense` layer whose output
reshapes to an $18 times 12 times 2$ spatial feature map. Three
depthwise-separable convolution blocks refine that map before a linear
$1 times 1$ head produces the final heatmap. Targets are Gaussian blobs
($sigma = 0.3$ m) at each person's position, decoded back by local-maximum
extraction with sub-cell centre-of-mass refinement and non-maximum suppression.

Capacity is spent almost entirely in the early fusion: 330,851 parameters, of
which 216,576 belong to the first `Dense` layer alone. Adding a second input
channel costs another 216,576 and a third exceeds the flash budget outright —
which is why Section 7's channel experiments trade against the radar count.

== Operator count: BatchNorm folding

The original blocks were `SeparableConv2D(activation="relu")` followed by
`BatchNormalization`. TFLite folds a BN into the preceding operator's weights
only when the path between them is purely linear, so a ReLU sitting in between
blocks the fold in both directions and three BNs shipped as runtime `MUL`+`ADD`
pairs. Reordering to `Linear → BatchNorm → ReLU`, with the nonlinearity as its
own layer, folds all three.

Separately, exporting with a *fixed batch-1 input signature* lets the converter
constant-fold the reshape. With a dynamic batch dimension Keras must build the
reshape target as `[tf.shape(x)[0], 18, 12, C]`, which ships as
`SHAPE → STRIDED_SLICE → PACK → RESHAPE` where one static `RESHAPE` would do.

Together these took the deployed graph from *20 operators to 11*, with no change
to the model itself.

= Quantization

#finding[
  *The calibration set matters more than anything else in the conversion, and it
  only became true after the BN fold.* Before folding, three live `MUL`+`ADD`
  pairs rescaled activations at every inference, and that rescaling silently
  absorbed a badly-chosen calibration range. Once BN folds into the convolution
  weights, that safety net is gone: the range picked at conversion time is the
  only thing setting int8 resolution.
]

The original pipeline calibrated on *raw*, non-decluttered frames — which had
been measured, pre-fold, to beat decluttered calibration. Post-fold, the same
choice was catastrophic: raw frames are far larger in magnitude than decluttered
ones, so the calibration range was set far too wide, and the quantized model
produced *5,569 false positives* on a routine post-conversion check. Switching
to decluttered calibration frames — the distribution the model actually trains
and deploys on — resolved it.

This is the clearest case in the project of an optimization being validated by
the wrong metric: the operator-count work succeeded on its own terms and would
have shipped a badly broken model if the check had stopped at "op count went
down".

= Validation methodology

== Grouped k-fold, never frame-level

At 25 Hz, adjacent frames are near-duplicates. A frame-level split therefore
leaks almost perfectly and inflates $F_1$ without measuring anything. All
validation is by *whole acquisition window*: six folds, stratified by the number
of people present, with each window scored by a model that never trained on it,
and with the per-fold $p_99$ and energy gate computed on that fold's training
windows only.

Detections are pooled across every fold and scored once, micro-averaged over the
full set of windows — never averaged per window.

== Reporting the interval, not the point

A leave-one-window-out jackknife over the 24 windows puts the standard error of
pooled $F_1$ at *± 0.018*. Several changes considered during the project moved
$F_1$ by less than a fifth of that.

For A/B comparisons the relevant quantity is smaller: two models scored on the
same windows form a *paired* comparison, per-window difficulty cancels, and the
SE of the difference is roughly half as large — ± 0.0089 for pooled $F_1$,
± 0.0183 for seated recall, and ± 0.0047 for walking recall. We adopt a change
only when its effect exceeds one paired SE, compared at *matched operating
points*: an intervention scored at its own tuned threshold against a baseline at
an untuned one is not a measurement.

== Why there is no lockbox

Holding out three windows would carry $"SE" approx 0.018 sqrt(24 slash 3)
approx 0.05$ — far too coarse to adjudicate the ~0.01-scale decisions actually
being made, and it would guard the wrong risk. All 24 windows are one
acquisition session; the real test is a different one, so a same-session holdout
resembles training far more than it resembles the test. The 12.5% of data would
buy false confidence. We report the interval and hold the adoption bar instead.

= Conditional inference

A per-frame model spends the same computation on an empty room as on four
walking people, but after decluttering, an empty room is near-silent. We use
this: the mean of the normalized magnitude over all 846 input cells is a single
scalar that distinguishes an occupied frame from an empty one. When it falls
below a calibrated gate, the model is not invoked at all, and the all-zero
heatmap it would have produced is substituted directly. The gate itself costs
846 additions, against roughly 330,000 multiply-accumulates for a full
inference — a skipped frame costs about 0.3% of the compute a normal one would.

Calibrating this gate produced two findings.

#finding[
  *The statistic that separates classes better makes a worse gate.* Reducing
  each frame to its *maximum* rather than its mean more than doubles the pooled
  separability ($d' = 5.53$ against 2.76) — but at a matched safety margin, it
  skips only 77.0% of empty-room frames, against 95.1% for the mean. The
  maximum's higher $d'$ comes from walking frames saturating at 1.0, which
  reflects how bright a walker is, not where the empty-versus-occupied boundary
  actually sits. A statistic measured far from the decision boundary does not
  predict behaviour at it.
]

#finding[
  *"Never skip an occupied frame" is not a calibratable rule on this dataset.*
  Window 021 holds a single seated subject with nobody else moving, and its
  occupied frames are *dimmer than an empty room* — median frame energy 0.01693
  against 0.01627 and 0.01632 for the two genuinely empty recordings, a ratio of
  1.04. A gate that protects it must sit below every empty room as well, and
  then skips 0.06% of frames: the feature buys nothing.
]

The gate is therefore calibrated against the windows where a person is separable
from an empty room *at all* — operationally, those whose 1st-percentile occupied
energy exceeds the pooled empty-room median — and set to a 0.9 margin below the
dimmest such frame. Window 021 is excluded by that measurement rather than by
name, so the rule survives a dataset in which a different window is the
pathological one. The result skips *95.1% of empty-room frames and 0.0000% of
occupied frames across all 21 separable windows*.

Frame 0 is always gated, in every recording, and correctly so: the decluttering
EMA seeds from it, so the decluttered frame 0 is identically zero and the model
has nothing to see there either.

This gate is not only a compute saving. Because it suppresses the model
precisely where phantom detections are most likely — a genuinely empty
room — it functions as a quality intervention as well.

= The seated-subject investigation

The single largest error mode in the project: eight of the 24 windows contain a
seated subject, and they hold roughly three quarters of all pooled false
negatives. Window 021, seated-only, scored $F_1 approx 0.006$.

== The signal is present; the per-frame representation is what fails

Within the decluttered complex CIR, energy in the 0.15–0.6 Hz respiration band
is 52–168× the energy at other frequencies in every window containing at least
one person — against only 1.8–1.9× in the two genuinely empty recordings. A
respiration signal is therefore measurably present whenever anyone is in the
room, seated or not. But a *single magnitude frame* separates seated-from-empty
at only $d' = 1.44$, against $d' = 2.45$ for the envelope of the same band
measured over time. What is missing is temporal context, not phase.

== Eight attempts, all negative

#table(
  columns: (auto, 1.1fr, 1fr),
  inset: 5pt,
  align: (left, left, left),
  stroke: 0.4pt + luma(180),
  table.header([*Channel set*], [*Idea*], [*Result*]),
  [`mag+delta32`], [complex difference at a 1.28 s lag], [pooled $F_1$ +0.6 SE (noise)],
  [`mag+aoa`], [phase difference between radars], [pooled $F_1$ −2.4 SE (regression)],
  [`iq`], [raw $I/Q$ instead of magnitude], [pooled $F_1$ −0.1 SE (noise)],
  [`mag+mag27+55+82`], [four instants across one breath], [$F_1$ −9.1 SE vs. 6 radars; −0.5 SE once radar count was controlled for],
  [oversampling ×3], [seated windows repeated in training], [pooled $F_1$ −1.7 SE; seated recall +0.8 SE, below the adoption bar],
  [`mag+breath`], [causal 0.15–0.6 Hz respiration bandpass], [seated recall +1.2 to +2.1 SE; walking recall −5.3 to −10.4 SE],
  [`mag+perio` (raw)], [periodicity at the respiration lag], [ruled out before training by its own pre-run check],
  [`mag+perio` (coherence)], [amplitude-invariant form of the above], [pooled $F_1$ −8 to −11 SE, the largest regression],
)

The two periodicity-based channels are described together because they failed
for *opposite* reasons, and together close off the approach rather than simply
adding one more negative result. In its raw form the statistic is mathematically
equal to $|A|^2 times "coherence"$ — reflector strength squared, times how
periodic the signal is — so it behaves as an energy measurement more than a
periodicity one, and a walker's much larger reflection amplitude beats a
breather's genuine periodicity by roughly 400×. Dividing out that amplitude
fixes this, but removes the one thing that let the statistic recognize an empty
cell: even pure noise, averaged the same way over a finite window, does not
settle to zero — it settles near $1 slash sqrt(N_"eff")$, a small but real
residual. Every empty cell (845 of the 846 in any given frame) then reads as
0.302 out of 1.0, rather than plain magnitude's near-zero 0.018, and the
early-fusion `Dense` layer sees this elevated background everywhere. This
version produced the largest regression of the entire investigation, including
on seated recall, the thing it was meant to fix. Since normalizing is a binary
choice and both branches were tried and failed for these opposite reasons, this
line of investigation is closed.

#finding[
  *A per-cell $d'$ is a feasibility check, not a prediction.* The coherence
  channel delivered 177.9 int8 levels of seated contrast against plain
  magnitude's 7.4 — a 24× better per-cell feature — and lost by 8–11 paired SE.
  Every gate it was given passed, several spectacularly. The number that
  predicted its failure was printed by the same gate table and not thresholded
  on. A printed number nobody asserts on is documentation, not a gate.
]

== What did work: the decode, not the input

Every approach above tried to make a static body visible within a single frame.
What actually worked instead operates on the model's own output, using a
property none of the input-side attempts could exploit: a seated person occupies
the same grid cell for the entire recording, while a spurious detection does
not.

The heatmap is smoothed with a causal EMA ($tau = 10$ s) and decoded as
$max(P, S dot theta slash t_"lo")$ at the unchanged threshold $theta$, so a cell
fires if it is bright now *or* has averaged above $t_"lo"$ for a while. The
rescaling lets one threshold decode both cases on a single surface, which keeps
the NMS and centre-of-mass steps seeing a coherent peak. It costs one float32
state array of 864 bytes and no retraining.

The diagnosis behind this splits the "seated problem" into two distinct causes.
For each seated person we can ask whether the per-frame decode recalled them,
and separately, how their own grid cell ranked among all 216 cells of that
frame's heatmap — a rank near the top means the model's raw output already
pointed at the right place, even though the decode's threshold discarded it:

#table(
  columns: (1fr, auto, auto),
  inset: 5pt,
  align: (left, center, center),
  stroke: 0.4pt + luma(180),
  table.header([*Condition*], [*Seated recall*], [*Rank of the seated cell of 216*]),
  [walker 1.5–2.0 m away], [64.6%], [median 15],
  [walker 2.0–3.0 m away], [64.2%], [(chance would be 108)],
  [walker 3.0–9.0 m away], [60.0%], [—],
  [no walker in the room (w021)], [0.6%], [median 203 — bottom 6%],
)

In *mixed* windows the model does place real evidence at the seated cell and the
per-frame decode was discarding it; that is 76% of all seated misses, and it is
what persistence recovers. In w021 the cell ranks below 94% of the room — there
is nothing to decode, and no output-side method can reach it.

#finding[
  *This is a genuine trade, not a free improvement.* Persistence gains +0.0184
  $F_1$ on the eight seated-containing windows and costs 0.0026 on the fourteen
  walking-only ones. The net pooled gain therefore depends on the proportion of
  seated-subject recordings resembling that of the training data (roughly one
  third of windows here); a session with substantially fewer such occurrences
  would see a correspondingly smaller benefit. An obvious refinement — gating
  the promotion on how steady a cell's activation is, to distinguish a
  stationary subject from a moving person's decaying trail — was tried, and
  restores walking-only $F_1$ exactly as fast as it removes the seated gain.
]

= Results

#finding[
  *Cross-validation $F_1$: 0.8682 ± 0.018 (1 SE), threshold 0.50, scored on the
  deployed int8 path with conditional inference and persistence both active.*
  This is the first out-of-fold measurement taken on int8 output rather than
  float Keras output — the two paths peak at different thresholds (int8 0.50,
  float 0.40) and disagree by +0.0145 $F_1$ at their respective optima. Measured
  seated recall is 68.6%, consistent with the 68.1% figure obtained
  independently on a separate machine.

  *Constraints:* 349.5 KB / 800 KB flash (44%), 11 operators, no forbidden ops,
  full INT8 — 5/5 checks pass. The activation-arena estimate reads *26–57 KB of
  the 300 KB budget* depending on the machine: it is a heuristic over the
  interpreter's tensor plan and moves with the TensorFlow build. Three runs on
  each of our two machines gave 56.6–57.4 KB and 26.2–27.0 KB (TF 2.20.0) for
  the same model file. Either reading leaves a 5–11× margin, and the flash
  figure reproduces identically on both.

  *Conditional inference:* 95.1% of empty-room frames skipped, 0.0000% of
  occupied frames skipped across every window where a subject is separable
  from empty at all, 11.5% of all frames pooled over the 24-window set.

  *Post-processing:* a full grid search (1008 configurations) over the decode
  threshold, NMS suppression radius, and tracker smoothing/coasting was run
  against this cache. It found a configuration reaching $F_1 = 0.8820$, +0.0138
  over the deployed one — but that result sits on the boundary of the searched
  threshold range and was not verified with a paired SE against the deployed
  configuration. We therefore kept the original constants ($theta = 0.50$,
  tracker $alpha = 0.4$, `max_distance` $= 1.0$, `max_coast` $= 2$) rather than
  adopt an unverified result.
]

#figure(
  image("images/predictions_vs_gt.png", width: 88%),
  caption: [Decoded occupancy against ground-truth positions. Peaks are refined
    to sub-cell precision by the centre of mass of their $3 times 3$
    neighbourhood before non-maximum suppression.],
)

= A documented negative: runtime layer skipping

Dynamic depth — skipping layers per frame according to input difficulty — is the
natural extension of Section 6 from the frame level into the graph, and it
cannot ship here. TFLM plans its activation arena *statically* at
`allocate_tensors()` and forbids `CUSTOM` operators, so there is no supported
way to vary the executed subgraph at runtime. Conditional inference therefore
delivers its saving entirely outside the graph, in the host code that decides
whether to invoke the interpreter at all — which violates nothing and needs no
operator support.

= Conclusion

The system meets every hard constraint with a wide margin and reaches its
accuracy through a small number of changes that were each measured against a
leave-window-out estimate and an explicit adoption bar. The two findings most
worth carrying forward are methodological rather than architectural: that an
optimization must be validated on the metric it can break, not the one it was
aimed at (Section 4), and that a summary statistic measured away from the
decision boundary — a per-cell $d'$, a pooled separability — repeatedly failed to
predict behaviour at it (Sections 6 and 7).
