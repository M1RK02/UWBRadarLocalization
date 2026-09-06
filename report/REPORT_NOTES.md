# Report notes — the working record behind `report.pdf`

This is the full experimental record: every finding, measurement and narrative
thread the report is built on, at a level of detail the report itself has no
room for. It is the evidence for the negative results in particular — the eight
falsified attempts on the seated-subject mode in Section 7 are each recorded
here with the numbers, the reasoning, and the mistakes made along the way.

Organized by topic, newest work first within each topic. Nothing here was
deleted once written; where a finding turned out wrong it is corrected in place
with the correction dated rather than erased — being wrong-then-corrected is
itself part of the story. The appendix at the end carries the measurement rules
all of it was produced under.

---

## The seated-subject investigation (Phase 4, `feat/seated-phase-channels`)

*Continued on `feat/seated-oversample` — Step 6 — `feat/seated-breath` — Step 7
— and `feat/seated-periodicity` — Step 8. 2026-08-26/28. A rigorous negative
result: four successive hypotheses were measured, built, tested on **nine** full
CV runs plus one that was **cancelled by its own pre-run gate**, and falsified.
Kept in full because the next person to look at seated subjects should not
repeat any of this — and because "we formed a wrong hypothesis, measured why,
and corrected it" is a stronger report section than a straight success would
have been.*

> **Read Step 5 before proposing any further input-representation change,
> Step 6 before proposing any further re-weighting of the training data,
> Step 7 before proposing any further band-limited temporal feature, and
> Step 8 before proposing any further ABSOLUTE-magnitude feature of any kind.**
> Six input representations and one data-distribution intervention have now
> been tried against this error mode. None is adoptable. The first six each
> moved the model along a single axis — how eagerly it fires on a static
> scene — and none gave it the ability to tell a static-*occupied* cell from a
> static-*empty* one. Step 6 is where that becomes explicit, because it is the
> one intervention that slid **down** that axis instead of up.
>
> **Step 7 is the exception, and it is the one to read first.** A causal
> respiration bandpass *did* make static-occupied separable from static-empty
> — seated recall +1.2 to +2.1 paired SE, the first gain above the bar in
> Phase 4, and empty-room phantoms ÷229 — and it is the first candidate not
> strictly dominated by the decode threshold. It is still not adoptable,
> because the same channel is dominated by walking motion (measured: walking
> windows carry 5–13× more in-band energy than the seated one), so walking
> recall regresses 5–10 SE. The premise that a 0.15–0.6 Hz band isolates
> breathing is now **falsified on real data**: a person crossing a range bin
> is a 1–3 s broadband transient that lands inside that band.
>
> **Step 8 then followed Step 7's own instruction — key on periodicity, not on
> the band — and the periodicity premise SURVIVED while the estimator did not.**
> Coherence at the respiration lag separates breathers from walkers essentially
> perfectly (1.0000 vs 0.0000–0.24, measured). But an *unnormalized* conjugate
> product returns `|A|² × coherence`, so walking's amplitude advantage returns
> squared: 18.7–85.5× on real data against `breath`'s 5–13×, and only **0.8 int8
> levels** of seated signal survive p99 normalization (`breath` delivered 20.1,
> plain `mag` 10.6). It was killed by its own pre-run gate for ~4 minutes and
> **zero CV runs** — the first time this project cancelled a run instead of
> spending it. The generalized finding is the one that matters going forward:
> **walking's amplitude swamps any absolute-magnitude feature, whatever temporal
> structure is built on top of it.**
>
> **Step 9 built the amplitude-invariant form Step 8 prescribed — return the
> coherence itself — and it is the largest regression Phase 4 has measured.**
> Every pre-run gate passed, several of them spectacularly: walking finally read
> *below* a seated subject (0.24×, against 18.7–85.5× uncorrected), and the
> seated cell delivered **177.9 int8 levels** against `mag`'s 7.4. The CV run
> then came back **−8.1 to −10.9 SE on pooled F1, −9.5 to −12.5 SE on walking
> recall, and −3.5 to −5.3 SE on SEATED recall** — worse at the very thing it
> was built for. The reason is a number the gate printed and did not gate on:
> amplitude invariance means the statistic has no way to say *"nobody is here"*,
> so an empty cell's own noise self-correlates and arrives at **0.30 of full
> scale against `mag`'s 0.018** — a 17× nuisance pedestal across all 845 other
> cells. **This closes the periodicity/coherence line.** Do not open a tenth
> variant of it.

### The target

8 seated-containing windows (004, 008, 009, 010, 011, 014, 015, 021) hold
**73.2–74.7%** of all pooled false negatives (audit measured 73.2%, an
independent second training run reproduced 74.7%). Window 021 (seated-only,
no other person) scores F1 **0.001–0.006** across measurements — the single
largest error mode found anywhere in the project.

### Step 1 — is the signal even there? Yes, overwhelmingly

Respiration puts measurable 0.15–0.6 Hz energy into the decluttered complex
CIR at **52–168×** the out-of-band noise floor in every one of the 22
occupied windows, against **1.8–1.9×** in both empty rooms — perfect
separation, ~28× margin. Window 021 carries the **highest** max SNR in the
entire dataset. Reproduced independently, exact numbers, via
`src/helpers/seated_diagnostics.py --quick`.

### Step 2 — why doesn't the model see it? Per-frame separability is weak

The model consumes one 40 ms magnitude frame at a time and is asked to
resolve a 0.305 Hz (3.28 s period) oscillation from it. Standardized
separability (d′ = |mean(A) − mean(B)| / pooled SD — Cohen's-d-style
detectability, on the subject's own best-SNR cell, empty room as the null):

| feature | d′ |
| --- | --- |
| magnitude, single frame (deployed today) | 1.436 |
| delta at lag 4 (160 ms) — the *original*, never-built pre-branch 4.2 idea | 0.699 |
| complex delta, lag 25 (1.0 s) | 1.664 |
| complex delta, lag 40 (1.6 s) | 1.941 |
| 0.15–0.6 Hz bandpass, complex (non-causal `filtfilt`, an upper bound) | 2.452 |

Per-frame magnitude alone is a weak one-shot signal — that part is real and
motivated building a temporal/phase feature.

### Step 3 — the hypothesis, and its falsification

**Hypothesis:** the mechanism is missing *time*, not missing *phase* — give
the model more temporal context and it recovers seated subjects. **Built and
tested, four full 6-fold CV runs (~22 min each):**

| channels | pooled F1 @0.40 | Δ vs `mag` | paired SE | verdict |
| --- | --- | --- | --- | --- |
| `mag` (baseline) | 0.8700 | — | — | deployed |
| `mag+delta32` | 0.8754 | +0.0053 | ±0.0089 | +0.6 SE — noise |
| `iq` | 0.8689 | −0.0012 | ±0.0086 | −0.1 SE — noise |
| `mag+aoa` | 0.8515 | **−0.0185** | ±0.0076 | **−2.4 SE — confirmed regression** |

**Important correction, made 2026-08-27:** `mag+delta32` uses lag **32 frames
= 1.28 s**, explicitly chosen (code comment) as "half a breath" — sitting
inside the productive zone the d′ table identifies. It is *not* the mistuned
160 ms / 4-frame idea (that one was never built as a real channel, only
measured in the exploratory diagnostic table above). This matters: the
ablation's failure is not "wrong lag." A well-tuned lag was tested and still
failed.

**No channel set recovered window 021.** F1 0.0109 (`mag`) → 0.0072
(`mag+delta32`) — slightly *worse*. Not merely weak: the subject's cell ranks
93/216 by mean response under `mag`, 10/216 under `mag+delta32`, but the
argmax of the mean heatmap still sits 1–2 m off in all four configurations.
The model does not localize a seated-alone subject at all, with any of the
four inputs tried.

**What `mag+delta32` actually turned out to be: a motion detector, not a
breathing detector.** Its one large, unambiguous, real effect is on the empty
room — phantom detections 3,124 → 423 (7.4× fewer), frames firing above 0.4 in
window 022: 263 → 10. It learned "nothing moving ⇒ nobody there," which
suppresses phantoms beautifully and is exactly backwards for a seated,
motionless subject. **Mechanistic reason it failed despite the right
timescale:** a 2-tap difference compares exactly two isolated samples 1.28 s
apart. If those two samples land at similar phases of the breathing cycle by
chance, the delta is small regardless of whether the person is breathing
normally — it is a coin-flip against oscillation phase, not a measurement of
it. This is different from — and cruder than — the continuous bandpass
integration that produced the strongest d′ (2.452) in the table above, which
was never built as an actual channel, only measured offline.

**Why `mag+aoa` regressed:** inter-antenna phase difference is a *spatial*
(angle-of-arrival) signal, not a *temporal* one — a single instant's phase
relationship between antennas, unrelated to whether anything is oscillating
over time. It directly tested, and falsified, the inference that "the
antennas are 89% redundant after magnitude, so the phase between them must be
free signal" (from the earlier leakage/correctness audit) — redundant does not
mean recoverable via a single extra instantaneous channel.

> **CORRECTION, 2026-08-27 (found while building 4.1d): the `aoa` channel
> never computed an inter-*antenna* phase difference at all.** Its code is
> `np.angle(z * conj(np.roll(z, -1, axis=-3)))`, and on `z` of shape
> `(T, 6, 3, 47)` axis `-3` is the **radar** axis — antennas are `-2`. Verified
> empirically with a synthetic per-(radar, antenna) phase ramp: the channel
> returns the radar-to-radar phase gap, identical across all three antennas of
> a radar. So `mag+aoa` differenced the phase of two *physically separate
> sensors metres apart and not phase-coherent* — a quantity with no
> angle-of-arrival meaning, which explains the −2.4 SE regression at least as
> well as the reasoning above does. The paragraph above is therefore an
> explanation of an experiment that was never actually run.
>
> **The code has deliberately NOT been changed.** Fixing the axis would
> silently turn the recorded `mag+aoa` number into a measurement of something
> else. If inter-antenna phase is ever worth testing, it should be added as a
> *new* channel name with the correct axis and its own CV run, leaving this
> one as the historical record it is. Note that the conclusion "adopt nothing
> from 4.1" is unaffected — a real `aoa` channel is simply untested, not
> vindicated.

**Why `iq` (raw I/Q) failed:** per-frame absolute phase wraps every ~2 cm of
range at UWB wavelengths — a per-frame phase value is near-aliased noise
unless integrated over time, exactly consistent with the delta-lag findings.

### Step 4 — two more checks, 2026-08-27, that reframe the problem

**Is w021's fold actually undertrained on "seated"? No.** The specific fold
that validates window 021 trains on 20 other windows, including **5 of the
other 7 seated windows and both empty rooms**. Not a zero-exposure fold — it
has substantial seated-person training data and still fails on w021
specifically. Rules out "the model has never seen the concept."

**Is the failure unique to w021, or pervasive?** Per-person OOF recall,
matching each ground-truth identity (walking vs. seated, by position spread)
to detections separately, in every mixed seated window:

| window | scenario | walking recall | seated recall |
| --- | --- | --- | --- |
| 004 | 1 walking + 1 seated | 99.8% | **8.7%** |
| 008 | 3 walking + 1 seated | 82.0% | 59.2% |
| 009 | 3 walking + 1 seated | 85.7% | 52.8% |
| 010 | 2 walking + 2 seated | 94.7% | 55.8% |
| 011 | 2 walking + 2 seated | 95.1% | 55.0% |
| 014 | 2 walking + 1 seated | 89.8% | 33.2% |
| 015 | 1 walking + 2 seated | 69.2% | 50.7% |

**The walking person is detected almost perfectly in every window. The
seated person caps around 33–59% even with full training exposure, and
collapses to 9% in window 004 specifically** — a near-w021-level failure
hiding inside a window whose *pooled* F1 looked unremarkable, because the
walker's near-perfect recall dominates the average. This is the finding that
matters most for future work: **pooled F1, and even the seated-subset F1,
both hid this.** Any future attempt must score per-person recall.

> **CORRECTION, 2026-08-27: the absolute numbers in this table do not
> reproduce on the other machine, and should be read as one draw, not as
> constants.** Re-measured on Christian's machine against the deployed `mag`
> model, with the now-committed `src/helpers/oof_breakdown.py`:
>
> | window | walking (here) | walking (above) | seated (here) | seated (above) |
> | --- | --- | --- | --- | --- |
> | 004 | 99.7% | 99.8% | **33.1%** | **8.7%** |
> | 008 | 90.5% | 82.0% | 73.9% | 59.2% |
> | 009 | 93.4% | 85.7% | 81.6% | 52.8% |
> | 010 | 98.6% | 94.7% | 85.5% | 55.8% |
> | 011 | 98.7% | 95.1% | 77.0% | 55.0% |
> | 014 | 95.2% | 89.8% | 48.1% | 33.2% |
> | 015 | 75.6% | 69.2% | 50.5% | 50.7% |
> | 021 | — | — | 0.3% | ~0 |
>
> This is **not** a matching-rule difference. Three rules were measured on the
> same cache — "any detection within 1.0 m" (seated 60.3% aggregate),
> exclusive Hungarian (53.6%), and no-tracker (45.6%) — and none reproduces
> the original table: w015 matches the first, w008 the second, and w004
> matches none of them (its 8.7% is below even the lowest variant's 17.4%).
> The two runs are different trained models on different machines with
> different library versions, which K8 already documents as producing
> non-bit-identical artifacts.
>
> **The lesson is the load-bearing part, and it survives intact:** the walking
> subject is recalled near-perfectly and the seated one is not, in every mixed
> window, on both machines. What does *not* survive is any specific
> percentage. **Per-identity seated recall is unstable across retrains** —
> which is itself evidence that seated detection sits right at the model's
> decision boundary. Always compare paired, same-machine, same-tool runs, and
> quote the paired SE (4.1d measured ±0.028–0.037 on aggregate seated recall,
> ~3–4× the pooled-F1 SE).

### Step 5 — 4.1d: four magnitude snapshots across one breath, on four radars

*2026-08-27, on the second machine (Christian's). The fifth CV run.
**Negative, and not marginally so.** Adopt nothing.*

**The design, and why it was worth one more run.** Steps 3–4 ruled out the
2-tap delta on a mechanism argument, not a tuning one: `mag+delta32` used a
well-chosen 1.28 s lag and still failed, because comparing two isolated
samples is a coin flip against the phase of a 3.3 s oscillation rather than a
measurement of it. 4.1d replaces the hand-designed comparison with raw
material: `mag(t)`, `mag(t−27)`, `mag(t−55)`, `mag(t−82)` — four evenly spaced
instants spanning one 0.305 Hz respiration period — and lets the early-fusion
Dense learn whatever combination it wants. Four channels only fit the 800 KB
flash budget on four radars, so the radar count dropped to **1, 2, 4, 6**
(both short-wall radars plus one from each long-wall pair; the geometry check
below says this costs no coverage on any real seated position).

Model: **691,811 parameters**, matching the budget arithmetic exactly.

**Result — worse on everything that matters, at any threshold.** The decode
threshold's argmax moved (0.40 → 0.30), so both operating points are shown.
Paired delete-one jackknife over the 24 windows, against this machine's
deployed `mag` baseline:

| threshold | metric | 4.1d | `mag` baseline | Δ | paired SE | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 0.40 | pooled F1 | 0.7929 | 0.8768 | **−0.0839** | ±0.0092 | **−9.1 SE** |
| 0.40 | walking recall | 0.8307 | 0.9474 | **−0.1167** | ±0.0121 | **−9.6 SE** |
| 0.40 | seated recall | 0.5459 | 0.6026 | −0.0567 | ±0.0365 | −1.6 SE |
| 0.30 | pooled F1 | 0.8047 | 0.8713 | **−0.0666** | ±0.0101 | **−6.6 SE** |
| 0.30 | walking recall | 0.8841 | 0.9609 | **−0.0768** | ±0.0076 | **−10.1 SE** |
| 0.30 | seated recall | 0.6220 | 0.6750 | −0.0531 | ±0.0282 | −1.9 SE |

4.1d's own best pooled F1 is **0.8047 @ 0.30**, against the baseline's 0.8768
@ 0.40 — a 0.072 drop where the adoption bar is a *gain* of one paired SE
(±0.009). This is the largest regression any Phase 4 candidate has produced;
`mag+aoa`'s −0.0185 was previously the worst.

**Seated recall did not move in the intended direction.** It fell, by 1.6–1.9
paired SE — not significant on its own, but unambiguously not the improvement
the hypothesis predicted. Per window, at the deployed threshold 0.40:

| window | scenario | walking Δ | seated Δ |
| --- | --- | --- | --- |
| 004 | 1 walking + 1 seated | −0.0 | **−19.0** |
| 008 | 3 walking + 1 seated | −11.5 | −2.0 |
| 009 | 3 walking + 1 seated | −17.0 | **−24.5** |
| 010 | 2 walking + 2 seated | −9.7 | −10.4 |
| 011 | 2 walking + 2 seated | −4.0 | +2.5 |
| 014 | 2 walking + 1 seated | −10.9 | −7.3 |
| 015 | 1 walking + 2 seated | −8.0 | +3.0 |
| 021 | 1 seated (alone) | — | +0.3 |

Only three windows moved up on seated recall, by +0.3 to +3.0 points, all
inside noise. Four moved down, two of them by 19 and 24 points.

**Window 021 is still not localized.** Seated recall 0.3% → 0.6%; window F1
0.006 → 0.013; frames firing above 0.4 rose 13 → 42 out of 7500. The argmax
of the mean heatmap sits **3.69 m** from the subject (baseline: 4.27 m) — both
in a room corner, against a true position of (2.26, 3.26) m. Two different
input representations, the same non-answer.

**It became a motion detector — again.** At a matched threshold of 0.40,
empty-room phantom detections fell **2062 → 160 (12.9×)**, while pooled
precision fell 0.9109 → 0.8625 and recall fell 0.8452 → 0.7338. So it is not
simply "more conservative": it makes *more* false positives around actual
people and far fewer in an empty room. That is the same mechanism
`mag+delta32` showed (7.4× fewer phantoms) reached by a completely different
temporal encoding — strong evidence the mechanism is a property of *giving
this architecture temporal context at all*, not of how the context is encoded.
Attribution between the temporal channels and the radar reduction is not
separable in one run, so this is stated as a joint effect.

**Constraints — viable, which is the point worth recording.** Re-run on the
quantized 4.1d model, not remembered: file **702.0 KB** (< 800 KB), activation
arena **27.6 KB** (< 300 KB), **5/5 PASS, exit 0**, full INT8, and an op
composition *identical* to the deployed baseline's (2 RESHAPE, 2
FULLY_CONNECTED, 3 DEPTHWISE_CONV_2D, 4 CONV_2D). The 4-channel/4-radar input
costs flash and nothing else. So the budget reasoning in 4.1d's spec was
sound — the configuration was deployable and simply did not work.

**What this closes.** The bottleneck is not the input representation. Five
representations have now been tested — `mag`, `mag+delta32`, `iq`, `mag+aoa`,
and `mag+mag27+mag55+mag82` — spanning per-frame magnitude, per-frame complex,
spatial phase, a 2-tap temporal difference, and a 4-point temporal window.
None recovered a seated subject. 4.1d had *more* capacity than the baseline
(691,811 vs 330,851 params) and *more* temporal context than any predecessor,
and was the worst of the five. **4.1b (out-of-distribution / data) is now the
only Phase 4 hypothesis left standing.**

**Caveat, stated plainly:** 4.1d changed two variables at once (channels and
radar count) because the flash budget forced it, so this run cannot say
whether the regression is driven by the temporal channels, by dropping two
radars, or by both. It does not need to: every candidate explanation is a
reason not to adopt it. If anyone wants to separate them later, the cheap
control is 4 radars with `mag` only — one CV run, and the plumbing for it is
already merged (`--radars 1,2,4,6 --channels mag`).

### Step 6 — 4.1b: oversample the seated windows in training

*2026-08-28, on Christian's machine. The sixth and seventh CV runs. **Negative,
and it fails in the opposite direction from every predecessor**, which is the
part worth reading. Adopt nothing.*

**The hypothesis.** Steps 3–5 closed the input-representation branch: five
representations, none recovered a seated subject, and the two that added
temporal context both became motion detectors. That left 4.1b — the bottleneck
is not *which features* the network sees per frame but *how much gradient*
stationary-but-occupied examples ever get. The overwhelming majority of
"occupied" training signal comes from walking people (big, unambiguous
frame-to-frame signal); the overwhelming majority of "static" training signal
is genuinely empty background. A seated person is the rare exception to both
correlations, so the loss is never strongly rewarded for detecting one.

**The intervention, and the one variable it moves.** `--seated-oversample K`
repeats the 8 seated-containing windows K times inside **each fold's training
set only**, before the global shuffle, so the loss sees a seated frame K times
per epoch and everything else once. Run at **K=3**. Nothing else changed:
channel set `mag`, all 6 radars, same architecture (330,851 params), same MSE
loss, same seeds, same fold split.

**Leak safety, verified rather than argued.** The bug that would silently
invalidate this entire experiment is applying the oversampling before or
outside the per-fold split — a held-out window duplicated into its own fold's
training set would raise the CV score for a reason unrelated to the hypothesis,
and nothing in the run output would look wrong. `src/helpers/verify_oversample.py`
asserts, at **frame granularity** and against the real fold construction (not a
re-implementation of it), on all 6 folds: no held-out frame appears in
training, each seated training window contributes exactly K × its frame count
and each other window exactly 1 ×, the validation split is untouched, and
**p99 is bit-identical between K=1 and K=3** (it is computed on the distinct
windows *before* oversampling, so the knob cannot move the input normalization
as a side effect).

**The baseline was regenerated, not remembered.** Step 4's correction
established that per-person recall does not reproduce across machines, so the
`mag` baseline was retrained from scratch on this machine with today's code
immediately before the comparison. It reproduced **pooled F1 0.8768 @ 0.40**
and a **73.2%** seated-window share of pooled FN — both matching the
historically recorded figures — and its 24 OOF caches came out **byte-identical
to `logs/20260826-122746/oof`**, the pre-Phase-4-refactor run. So the channel-set
infrastructure, the radar-subset infrastructure and the K=1 oversampling path
are all confirmed bit-exact no-ops on the deployed configuration. That is a
reproducibility result worth keeping independently of this experiment's outcome.

**Result — the seated gain is below the bar, and it is paid for twice.** Both
models' threshold argmax is 0.40, so for once the matched operating point is
also each model's own best one and 4.1d's measurement trap does not apply.
Paired delete-one jackknife over the 24 windows:

| threshold | metric | K=3 | `mag` baseline | Δ | paired SE | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 0.40 | pooled F1 | 0.8605 | 0.8768 | **−0.0163** | ±0.0099 | **−1.7 SE** |
| 0.40 | walking recall | 0.9325 | 0.9474 | **−0.0149** | ±0.0089 | **−1.7 SE** |
| 0.40 | seated recall | 0.6409 | 0.6026 | +0.0384 | ±0.0510 | +0.8 SE — **under the bar** |
| 0.30 | pooled F1 | 0.8579 | 0.8713 | **−0.0134** | ±0.0071 | **−1.9 SE** |
| 0.30 | walking recall | 0.9524 | 0.9609 | **−0.0085** | ±0.0043 | **−2.0 SE** |
| 0.30 | seated recall | 0.6950 | 0.6750 | +0.0200 | ±0.0373 | +0.5 SE |

The adoption bar was a seated-recall gain of **more than 1 paired SE without
materially regressing walking recall or the phantom rate**. Seated recall moved
+0.8 SE at 0.40 and +0.5 SE at 0.30 — the right direction, for the first time in
Phase 4, but inside noise at both operating points — while walking recall
regressed 1.7–2.0 SE and pooled F1 regressed 1.7–1.9 SE. **Fails the bar on all
three clauses.**

**It became a motion detector in reverse — and that is the finding.** At a
matched 0.40, empty-room phantom detections went **2,062 → 11,321 (5.5× MORE)**;
at 0.30, 7,823 → 14,847 over 15,000 frames, i.e. it fires on essentially every
frame of an empty room. Per empty window at 0.40: w022 1,321 → 7,257, w023
741 → 4,064.

| intervention | empty-room phantoms vs. its baseline | seated recall Δ |
| --- | --- | --- |
| `mag+delta32` (Step 3) | **÷7.4** | ~0 |
| 4.1d temporal window (Step 5) | **÷12.9** | −1.6 to −1.9 SE |
| **4.1b oversampling (this step)** | **×5.5** | +0.5 to +0.8 SE |

The two temporal encodings learned *"nothing moved ⇒ nobody there."*
Oversampling taught the exact inverse — *"stationary ⇒ someone there"* — by
making stationary-occupied frames three times as common without making them any
more **distinguishable** from stationary-empty ones. Three interventions, two
mechanisms, one axis: how eagerly the model fires on a static scene. None of
them gave it the ability to discriminate a static-occupied cell from a
static-empty one, which is the thing that would actually have to change.

**The dominance check that settles adoption.** This is a cross-threshold
comparison, deliberately, and not a matched one — the question it answers is
not "which is better at a fixed operating point" but "did the intervention
produce anything the existing threshold knob could not":

| model @ threshold | seated recall | empty-room phantoms | pooled F1 |
| --- | --- | --- | --- |
| K=3 @ 0.40 | 64.09% | 11,321 | 0.8605 |
| **baseline @ 0.30** | **67.50%** | **7,823** | **0.8713** |

The baseline, simply by lowering its own decode threshold, reaches **higher**
seated recall with **fewer** phantoms and **higher** pooled F1 than the
oversampled model does. The oversampled model is strictly dominated on all
three metrics at once by a post-processing knob that costs no retraining. There
is nothing here to adopt even before the paired SEs are considered.

**Per-window seated recall is a wash of large opposing moves**, at 0.40:

| window | scenario | seated K=3 | seated baseline | Δ (pp) |
| --- | --- | --- | --- | --- |
| 004 | 1 walking + 1 seated | 18.0 | 33.1 | **−15.0** |
| 008 | 3 walking + 1 seated | 83.9 | 73.9 | +10.0 |
| 009 | 3 walking + 1 seated | 84.4 | 81.6 | +2.8 |
| 010 | 2 walking + 2 seated | 75.8 | 85.5 | **−9.7** |
| 011 | 2 walking + 2 seated | 84.8 | 77.0 | +7.8 |
| 014 | 2 walking + 1 seated | 56.2 | 48.1 | +8.1 |
| 015 | 1 walking + 2 seated | 70.5 | 50.5 | **+20.1** |
| 021 | 1 seated (alone) | 0.4 | 0.3 | +0.1 |

Five windows up, two down, one flat, and the aggregate +3.8 pp is carried
almost entirely by w015's +20.1. The ±0.0510 paired SE on aggregate seated
recall — larger than the ±0.028–0.037 that 4.1d measured, and ~5× the pooled-F1
SE — is the quantitative form of this scatter. Step 4's correction predicted
exactly this: per-identity seated recall sits at the model's decision boundary
and is unstable across retrains. **w004, the window whose seated subject was
already the second-worst in the dataset, got substantially worse under the
intervention designed to help it.**

**What it did do, and why it still is not a win.** The intervention moved errors
from the seated population to the walking one rather than removing them. At
0.40, the seated-containing windows' share of pooled FN fell **73.2% → 63.8%**
and the seated-subset F1 rose 0.8174 → 0.8280 — but walking-only FN rose
17,427 → 23,931 (**+37%**) and walking-only F1 fell 0.9244 → 0.9064. Pooled
precision fell 0.9109 → 0.8793. A redistribution, not a repair.

**Window 021 is still not localized.** Seated recall 0.3% → 0.4%; window F1
0.006 → 0.009; 33 true positives out of 7,500 frames, against 22 for the
baseline. Six interventions across two branches, the same non-answer. Whatever
w021 needs, neither the input representation nor the training distribution
supplies it.

**Caveats, stated plainly.**

1. **Oversampling by repetition also raises total gradient volume**, not only
   the seated:everything-else *ratio*. A fold's training set goes 150,000 →
   255,000 rows, so at a fixed epoch cap and `EarlyStopping` patience the K=3
   run takes ~1.7× more gradient steps per epoch. Holding total volume fixed
   would have required subsampling walking frames — a different second
   variable, not a cleaner experiment. The regression could in principle be
   partly an over-training effect rather than a re-weighting effect; every
   candidate explanation still argues against adopting it.
2. **Fold 3 holds out three seated windows at once** (008, 015, 021), so it
   reaches only 1.50× effective oversampling against 1.70× for the other five
   folds. The intervention is therefore weakest in precisely the fold that
   validates w021. This is inherent to grouped CV with 8 seated windows in 6
   folds, not a defect, but it means w021's non-response is measured under a
   milder dose than the headline K suggests.
3. **A bit-identical A/B is impossible here by construction.** The global
   permutation runs over a different-length array and there are more batches per
   epoch, so the radar-dropout draws necessarily differ. Same situation the
   4-radar run already documents.
4. **Only K=3 was run.** K=2 was not, on the reasoning that the measured trade
   is structural rather than mistuned: seated recall and phantom rate moved
   together along one axis, so a *smaller* K yields a smaller share of an
   already sub-bar gain, and a *larger* K pushes further into the walking and
   phantom regressions. Clearing a >1 SE seated bar (>5.1 pp) would require
   more dose than K=3, in the direction the evidence says makes everything else
   worse. That is an argument from the measured trade-off, not a proof of
   monotonicity — if anyone reopens this, K=2 is the cheap check.

**What this closes.** 4.1b was the last Phase 4 hypothesis standing after the
input-representation branch closed. Both branches are now measured and negative,
by the same standard. The seated failure is **neither** an input-representation
problem **nor** a gradient-volume problem — the model can be made arbitrarily
eager or arbitrarily reluctant to fire on a static scene, and at no setting of
either knob does it separate a static-occupied cell from a static-empty one.
Any future attempt has to attack that discrimination directly rather than
re-weighting or re-encoding what the model already sees.

> **UPDATE, 2026-08-28 (Step 7): the closing sentence above was the right
> instruction, and following it worked.** 4.1c attacked the discrimination
> directly — a causal respiration bandpass, integrating the band *before* the
> model sees a frame — and it **did** separate static-occupied from
> static-empty: seated recall +1.2 to +2.1 paired SE, above the bar at every
> matched threshold, with empty-room phantoms ÷229. So "the model cannot be
> given that discrimination" is **falsified**; what remains true is that it is
> not adoptable, for a different reason. The bandpass is dominated by walking
> motion (5–13× more in-band energy in walking windows than in the seated one),
> so walking recall pays 5–10 SE. Read Step 7 — the open problem is no longer
> "make static-occupied separable" but "separate respiration from broadband
> transients that share its band".

### Step 7 — 4.1c: a causal respiration bandpass as an input channel

*2026-08-28, on Christian's machine. The eighth and ninth CV runs.
**Negative — it fails the adoption bar on two of its three clauses — but it is
the first intervention in Phase 4 to clear the seated-recall bar, and it fails
for a reason none of the six before it did.** Adopt nothing; read the
mechanism, because it retires a premise the whole branch was built on.*

**Why this was reopened.** 4.1c was struck through as "superseded by 4.1d". It
was shelved, not falsified — 4.1d then failed by a well-understood mechanism of
its own, which left the original idea untested. The distinction it rests on is
real: `mag+delta32` (Step 3) and 4.1d (Step 5) both handed the network RAW
values from a few discrete instants — a 2-tap difference and four raw snapshots
— and let it find its own combination. Both times it found *"did anything
change"* (a motion detector) rather than *"is there 0.15–0.6 Hz energy here"*
(a breathing detector). This channel does the temporal integration **before**
the model sees a frame, so a stationary breather differs from an empty cell by
construction and there is no raw multi-timepoint comparison left in the input
to shortcut through.

**What was built.** One new primitive channel, `breath`, and one channel set,
`mag+breath` (`src/preprocessing.py`). All 6 radars, all 3 antennas, nothing
else changed. **547,427 parameters**, matching the budget arithmetic
(`114,275 + 6·2·3·47·256`) exactly. A causal Butterworth bandpass on the
**complex** decluttered CIR — filtering the magnitude would discard the phase
rotation that a sub-millimetre chest displacement actually produces — then
`sqrt(re²+im²)` rectification (the same convention as `mag`) and a one-pole
EMA. `lfilter` carrying its own `zi`, never `filtfilt`: Step 2's d′ 2.452 came
from a non-causal filter that reads the future, so it was never a deployable
number. Not in `SIGNED_CHANNELS` — a rectified envelope is non-negative.

#### Verification before the CV run, not after

`mag+aoa` cost a full CV run measuring the wrong axis and was caught weeks
later by a synthetic test rather than by review. So this channel got its
evidence first, as `src/helpers/verify_breath_channel.py` (committed, six
checks, all asserted). The actual printed output:

```
=== 1. frequency response ===
  butter(N=3, [0.15, 0.6] Hz, 'band', fs=25.0) -> filter order 6, 7 b / 7 a taps
  max pole radius 0.987741
  peak gain 1.0000 at 0.3051 Hz
  -3 dB band 0.1501 .. 0.5999 Hz   (target 0.15 .. 0.6)

   freq (Hz)   gain (dB)   note
        0.00    -6000.00   DC -- static clutter residue, must be rejected
        0.05      -35.38   3 breaths/min, below any real subject
        0.15       -3.01   band edge
        0.30       -0.00   band centre: w021's measured 0.305 Hz
        0.60       -3.01   band edge
        1.00      -18.51   fidgeting
        2.00      -38.79   walking / bulk motion
        5.00      -66.40   fast motion
       12.00     -146.86   near Nyquist

=== 2. synthetic separation ===
  1500 frames (60 s), amplitude 0.1 on both moving cells. Steady state = frames 500+ (20 s).

  cell                    (r,a,bin)        mean         min         max
  breathing 0.3 Hz       (0, 0, 10)    0.099997    0.099898    0.100037
  moving 2.0 Hz          (1, 1, 20)    0.001149    0.001132    0.001169
  empty (flat zero)      (2, 2, 30)    0.000000    0.000000    0.000000
  empty (noise)          (3, 0, 40)    0.000234    0.000069    0.000686

  breathing / moving   =       87.0x   (worst case, min/max = 85.5x)
  breathing / noise    =      426.5x
  flat-zero cell       = 0.000e+00 (max over steady state)
```

Also asserted: the output contract (`(T,6,3,47)`, float32, finite,
non-negative), and a bounded cold start — frame 0 reads `3.9e-14` and the peak
transient is `1.06×` the steady state, so the `zi` initialization degrades
gracefully with no explicit clamp, the same guarantee `_lagged` gets from
clamping to `z_0`.

**Two deviations from the spec as written, both forced by measurement.**

1. **A 10 s synthetic is too short to test this filter.** The 6-pole design has
   a 3.25 s time constant (poles at radius 0.9877), so at 250 frames it is
   still mid-transient. The test runs 1500 frames and measures steady state
   from frame 500.
2. **The dtype guidance was backwards, and following it would have silently
   destroyed the channel.** `lfilter_zi(b,a).reshape(...) * z[0:1]` yields
   **complex128**, not complex64 — under NEP 50, float64 × complex64 promotes.
   Forcing float32 coefficients and a complex64 state makes this filter
   **diverge to NaN at frame 3743, ~150 s into a 300 s window**. The upcast is
   load-bearing, not an accident to optimize away. Check 6 of the verifier
   asserts the divergence so nobody "fixes" the dtype later. Any deployment
   must use an `sosfilt` biquad cascade, which is stable in float32 to a
   2.2e-5 relative error.

**Real SRAM cost, which `evaluate_constraint.py` cannot see:** 3 biquad
sections × 2 states × (re, im) × 4 B + one EMA state, over all 846 cells =
**43.0 KB — 4× SMALLER than 4.1d's 183 KB ring buffer.** A recursive filter
carries 6 numbers per cell however far back it integrates; a ring buffer
carries one per frame of history. That is the structural advantage of putting
the integration in the filter, and it is the one part of this experiment that
came out strictly better than its predecessor.

**Feasibility d′ (a feasibility check, never a prediction — discipline #9):**
on w021's best-SNR cell against the empty room, `mag` reproduced Step 2's
**1.436** exactly and the causal `breath` channel read **3.643**. That is
*higher* than Step 2's non-causal "upper bound" of 2.452, which needs
explaining rather than celebrating. Decomposed: the causal filter alone scores
2.449 (so causality costs nothing here), and the entire 2.45 → 3.65 gain is the
10-frame EMA cutting variance — an axis Step 2's measurement never varied. The
2.452 was an upper bound for an *unsmoothed* feature, not for all bandpass
features.

#### Measurement

Two full 6-fold CV runs. The `mag` baseline was **retrained from scratch on
this machine immediately before the comparison** (discipline #10) rather than
reusing Step 6's recorded number. It reproduced **pooled F1 0.8768 @ 0.40**,
the **73.2%** seated-window share of pooled FN, and precision 0.9109 / recall
0.8452 — all exactly — and its 24 OOF caches came out **byte-identical to
`logs/20260828-125401`**. So adding the `breath` channel is a confirmed
bit-exact no-op on the deployed `mag` path, and every fold's `mag` p99 is
unchanged (fold 1: 317.8705 in both runs).

`mag+breath`'s threshold argmax moved to **0.30** against the baseline's
**0.40** — 4.1d's measurement trap exactly (discipline #7) — so every
comparison below is at a matched threshold, and the whole matched range is
reported rather than one convenient point.

| threshold | metric | `mag+breath` | `mag` baseline | Δ | paired SE | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 0.40 | pooled F1 | 0.8500 | 0.8768 | **−0.0268** | ±0.0102 | **−2.6 SE** |
| 0.40 | walking recall | 0.8694 | 0.9474 | **−0.0780** | ±0.0075 | **−10.4 SE** |
| 0.40 | seated recall | 0.6418 | 0.6026 | **+0.0392** | ±0.0330 | **+1.2 SE** |
| 0.30 | pooled F1 | 0.8648 | 0.8713 | −0.0065 | ±0.0113 | −0.6 SE |
| 0.30 | walking recall | 0.9176 | 0.9609 | **−0.0433** | ±0.0055 | **−7.9 SE** |
| 0.30 | seated recall | 0.7112 | 0.6750 | **+0.0362** | ±0.0279 | **+1.3 SE** |
| 0.20 | pooled F1 | 0.8574 | 0.8552 | +0.0022 | ±0.0137 | +0.2 SE |
| 0.20 | walking recall | 0.9362 | 0.9649 | **−0.0287** | ±0.0046 | **−6.2 SE** |
| 0.20 | seated recall | 0.7644 | 0.7268 | **+0.0376** | ±0.0249 | **+1.5 SE** |
| 0.15 | pooled F1 | 0.8457 | 0.8443 | +0.0014 | ±0.0142 | +0.1 SE |
| 0.15 | walking recall | 0.9396 | 0.9653 | **−0.0257** | ±0.0045 | **−5.7 SE** |
| 0.15 | seated recall | 0.7852 | 0.7439 | **+0.0413** | ±0.0230 | **+1.8 SE** |
| 0.10 | pooled F1 | 0.8239 | 0.8280 | −0.0040 | ±0.0150 | −0.3 SE |
| 0.10 | walking recall | 0.9401 | 0.9652 | **−0.0251** | ±0.0048 | **−5.3 SE** |
| 0.10 | seated recall | 0.8075 | 0.7572 | **+0.0504** | ±0.0236 | **+2.1 SE** |

**The seated bar is cleared, at every matched threshold, for the first time in
Phase 4.** +1.2 to +2.1 paired SE, rising as the threshold falls. Every prior
attempt was at best +0.8 SE (Step 6) and usually negative. This is a real
effect, not a threshold artifact — it holds across the whole operating range.

**And it is paid for with the walking population, at 5–10 SE.** That is not a
marginal trade: walking recall regresses by more than *four times* the seated
gain in SE terms, at every threshold. The adoption bar was a seated gain of
>1 paired SE **without materially regressing walking recall** and **without
the phantom-rate trade-off**. Clause 1 passes; clauses 2 and 3 fail.

**Empty-room phantoms collapsed harder than in any previous run.** At a matched
0.40, **2,062 → 9 detections over 15,000 frames — 229× fewer**; at 0.30,
7,823 → 12 (652×). For comparison, `mag+delta32` was ÷7.4 and 4.1d was ÷12.9.

| intervention | empty-room phantoms vs. its baseline | seated recall Δ |
| --- | --- | --- |
| `mag+delta32` (Step 3) | ÷7.4 | ~0 |
| 4.1d temporal window (Step 5) | ÷12.9 | −1.6 to −1.9 SE |
| 4.1b oversampling (Step 6) | ×5.5 | +0.5 to +0.8 SE |
| **`mag+breath` (this step)** | **÷229** | **+1.2 to +2.1 SE** |

**The specific check this experiment existed to make.** The question was
whether w021 improves *without* the empty-window phantom rate moving materially
in either direction — both better, not one traded for the other, which is what
"static-occupied became distinguishable from static-empty" would actually look
like. **The answer is no, on both halves.** w021's seated recall is 0.16% vs
the baseline's 0.29% at 0.40, and 0.55% vs 0.51% at 0.30 — unchanged or
slightly worse, the seventh intervention to leave it unlocalized. And the
phantom rate moved by 229×, the largest move of the four. The one place w021
does respond is at a threshold of 0.10, where it reaches **11.97% against the
baseline's 0.75%** with only 76 phantoms against 15,094 — a real 16× move, but
at an operating point whose pooled F1 (0.8239) is 0.05 below the deployed one,
and still 88% of the window undetected.

**Dominance check (discipline #12), and here the answer differs from Step 6.**

| model @ threshold | seated recall | walking recall | empty-room phantoms | pooled F1 |
| --- | --- | --- | --- | --- |
| `mag+breath` @ 0.30 | 71.12% | 91.76% | **12** | 0.8648 |
| baseline @ 0.30 | 67.50% | 96.09% | 7,823 | 0.8713 |
| baseline @ 0.20 | 72.68% | 96.49% | 13,733 | 0.8552 |

Step 6's oversampled model was *strictly dominated* — the baseline at a lower
threshold beat it on seated recall, phantoms and F1 at once. **This one is
not.** No baseline threshold reaches `mag+breath`'s seated recall and its
phantom count together: matching the seated recall costs th ≤ 0.20, which
brings 13,733 phantoms against 12. So this channel does buy something free
post-processing cannot — near-zero empty-room false alarms at a competitive
seated recall. It still fails the bar, on walking recall. Both things are true
and the report should say both.

**A redistribution again, and a larger one.** At 0.40 the seated-containing
windows' share of pooled FN fell **73.2% → 59.7%**, but walking-only FN rose
**17,427 → 36,617 (+110%)** and total FN rose 65,028 → 90,874 (+40%).
Precision rose 0.9109 → 0.9288. Per-window seated recall at 0.40 is again a
scatter of opposing moves — w014 **+17.0 pp**, w015 **+13.6 pp**, w008 +6.8,
w010 +3.9, w011 +3.4, against w009 **−13.7 pp**, w004 **−8.7 pp**, w021 −0.1
— with the aggregate carried by w014 and w015. w004, the second-worst seated
identity in the dataset, got worse under a third consecutive intervention
meant to help it.

#### The mechanism, measured — and it retires a premise

The obvious explanation for the walking collapse is that a 0.15–0.6 Hz
bandpass rejects walking motion (−38.8 dB at 2 Hz), so the channel is ~0 for a
walker and the model learned to gate on it. **That explanation is wrong, and
measuring it is the most useful thing this run produced.** Mean per-frame peak
of the `breath` channel, by scenario:

| window | scenario | `breath` peak | `mag` peak | breath/mag |
| --- | --- | --- | --- | --- |
| 016 | 1 walking | 94.18 | 601.19 | 0.157 |
| 005 | 4 walking | 258.23 | 1651.95 | 0.156 |
| 004 | 1 walking + 1 seated | 119.49 | 711.38 | 0.168 |
| **021** | **1 seated** | **20.08** | **41.04** | **0.489** |
| 022 | empty | 6.36 | 36.81 | 0.173 |

**Walking windows carry 5–13× MORE absolute in-band energy than the seated
window.** The channel is not a breathing detector on real data; it is an
*in-band energy* detector, and walking dominates it. The reason is that the
synthetic test's premise does not hold outside the synthetic test: real walking
is not a 2 Hz sinusoid to be rejected, it is a **broadband transient** — a
person crossing a 0.15 m range bin produces a pulse lasting 1–3 s, whose
spectrum sits squarely inside the respiration band. Only 2–9% of a walking
cell's power is in-band, but 2% of a walker's enormous total dwarfs 100% of a
breather's. Walking also saturates the channel: at the fold p99, 2.1% of cells
clip in the 4-walker window against 0.002% in w021.

What the channel *does* deliver is the ratio: w021's breath/mag of 0.489
against an empty room's 0.173 is a genuine 2.9× contrast that a single
magnitude frame does not have — which is precisely why seated recall rose and
phantoms collapsed. The band separates *stationary-occupied from
stationary-empty*, as designed. It does not separate *breathing from walking*,
which nobody checked because the synthetic test made it look as though it did.

**The lesson, and it is the transferable one: a synthetic test with pure tones
validates the filter, not the premise that the world is made of pure tones.**
The verification was correct and worth doing — it caught a NaN divergence that
would have silently corrupted the channel — and it still could not have
predicted this, because the failure is in the modelling assumption rather than
in the implementation. The right extra check, cheap and skipped, was to run the
finished channel over one real walking window and one real seated window and
compare the two distributions *before* spending two CV runs.

**Caveats, stated plainly.**

1. **Two variables move together, unavoidably.** `mag+breath` has 547,427
   parameters against the baseline's 330,851, so this run cannot separate "the
   breath channel hurts walkers" from "a second early-fusion channel
   destabilizes the walking fit". 4.1d saw a near-identical walking regression
   (−9.6 SE at 0.40) with a *different* second channel and a different radar
   count, which is weak evidence for the second explanation. The cheap
   separating control, if anyone wants it, is a 2-channel set whose extra
   channel is known-inert.
2. **Only one filter configuration was run.** Order 3, band 0.15–0.6 Hz,
   EMA α=0.9. The d′ decomposition above says the smoothing carries the whole
   feasibility gain, so α is the parameter most likely to be mistuned — but
   the measured trade is structural (seated and walking moved in opposite
   directions at *every* threshold), so tuning α buys a share of a gain that is
   already paid for on the walking side.
3. **The phantom collapse is not automatically good.** An empty-room phantom
   rate of 9/15,000 looks like a triumph in isolation; read together with
   walking recall at 86.94%, it is the same "more conservative overall" move
   that Steps 3 and 5 made, reached by a third mechanism.
4. **w021's response at threshold 0.10 (11.97% vs 0.75%) is one draw on an
   unstable metric** (discipline #8) and sits at an operating point nobody
   would deploy. It is recorded because it is the first non-zero movement that
   window has shown in seven interventions, not because it is a result.

**What this closes.** 4.1c is now run rather than superseded, and Phase 4's
input-representation branch has been tested six ways: `mag`, `mag+delta32`,
`iq`, `mag+aoa`, `mag+mag27+mag55+mag82`, and `mag+breath`. The reopening was
justified — the mechanism was genuinely untested and it produced a genuinely
different result, the first seated gain above 1 SE and the first candidate not
strictly dominated by the threshold knob. It is still not adoptable, and the
reason is now specific rather than general: **pre-integrating the respiration
band does make static-occupied separable from static-empty — the thing Steps
3–6 said would have to change — and that separation is bought by a channel
that walking motion dominates, so it costs more on the walking population than
it returns on the seated one.** Any successor has to isolate respiration from
*broadband transient* energy, not merely from high frequencies. A band-limited
linear filter cannot do that, because the two overlap in frequency; it needs
something that keys on the *periodicity* of the signal rather than its band —
autocorrelation, spectral flatness within the band, or a persistence
requirement over several breath cycles.

### Step 8 — 4.1e: a periodicity channel, killed by its own gate before any CV run

*2026-08-28, on Christian's machine. **Zero CV runs.** Negative, caught by the
pre-run measurement Step 7's lesson mandates — which is the point, and the
cheapest negative result in the project: ~4 minutes of measurement against the
~45 minutes a CV run costs. Adopt nothing.*

**The hypothesis, and why it was the right successor.** Step 7 closed with a
specific instruction: respiration and broadband walking transients overlap in
*frequency*, so no band-limited linear filter separates them, and a successor
has to key on **periodicity** — "does this cell repeat at the respiration lag"
rather than "how much energy is in this band". A one-off transient should fail a
repetition test even when it has in-band energy, because its own lagged self is
empty; a genuine oscillation should pass it every cycle. That reasoning is
sound, and the mechanism built from it works exactly as designed. It still fails,
for a reason neither Step 7 nor the design brief anticipated.

**What was built.** One primitive channel, `perio`, and one channel set,
`mag+perio` (`src/preprocessing.py`). At each of three candidate lags — 60, 82,
100 frames = 0.417 / 0.305 / 0.250 Hz, spanning the resting adult range — take
the conjugate product `z(t) · conj(z(t−lag))`, smooth it causally with a one-pole
EMA whose time constant is `3·lag` (so each candidate gets ~3 of *its own*
cycles of averaging rather than one shared frame count), take the magnitude, and
reduce across the three lags with a per-cell, per-frame max. All 6 radars, all 3
antennas, nothing else changed. Not in `SIGNED_CHANNELS` — a rectified magnitude.
Confirmed a **bit-exact no-op on the deployed `mag` path** (identical sha256 over
windows 0/4/16/21/22 before and after the change).

#### The gate, and it is the whole result

`src/helpers/verify_perio_channel.py`, run **before** any training. It exits 1
when either gate fails, so it cannot be walked past silently.

**1. The synthetic mechanism check — the design premise holds.** The response
decomposes exactly as `|A|² × coherence`, so both factors are printed:

```
  cell                         kind           response      |z|^2  coherence  vs breather
  breathing, lag 82 (0.305 Hz) periodic       0.010000   0.010000     1.0000          1.0x
  breathing, lag 70 (0.357 Hz) periodic       0.010000   0.010000     1.0000          1.0x
  breathing, lag 91 (0.275 Hz) periodic       0.010000   0.010000     1.0000          1.0x
  transient 1 s, x20 amp       transient      0.000000   0.017725     0.0000          0.0x
  transient 3 s, x20 amp       transient      0.004100   0.053174     0.0771          0.4x
  transient 3 s, x1 amp        transient      0.000010   0.000133     0.0771          0.0x
  gait 2 Hz burst, x20 amp     adversarial    0.016807   0.070898     0.2371          1.7x
  serpentine, re-crosses/4 s   adversarial    0.707787   0.695602     1.0175         70.8x
  empty (flat zero)            empty          0.000000   0.000000     0.0000          0.0x
  empty (noise)                empty          0.000000   0.000002     0.0697          0.0x
```

Read the **coherence** column: every breather scores exactly 1.0000 and every
one-off walker scores 0.0000–0.24. *The discrimination the design was built to
produce is real and near-perfect.* A 1 s crossing carrying **20× the amplitude**
reads exactly **0.000000**, because at lag ≥ 60 frames (2.4 s) a 1–3 s crossing
barely overlaps its own lagged self.

Two other premises were confirmed here rather than assumed. **The fixed-lag
premise holds completely**: breathers at 0.357 and 0.275 Hz, deliberately placed
*between* the candidate lags, keep **1.000×** the on-lag response — a
mismatched-but-consistent lag settles at a fixed phase offset and the magnitude
discards it, exactly as the design brief argued. And the **cold start** is
monotonic with no overshoot (frame 0 = 0.0, no clamp needed).

**And the scaling defeats the mechanism anyway.** The channel returns `|A|² ×
coherence`, not `coherence`. A walker's `|A|` is ~20× a breather's, so its `|A|²`
is ~400×, and weak coherence on a huge amplitude beats perfect coherence on a
tiny one. A 3 s crossing at 20× amplitude reaches 0.41× a breather on coherence
0.077 alone; a walker who re-crosses the same cell every 4 s reaches **70.8×**.

**2. The real-data gate — discipline #13, applied literally.** Ground-truth
positions were mapped to radar cells using a range-bin offset **calibrated from
the data on the `mag` channel only** (so the calibration cannot flatter the
channel under test). Two independent estimators agree: median(argmax bin −
nominal bin) = **+6**, and the argmax of the mean-magnitude-vs-offset profile =
**+6** (+0.90 m), which also independently confirms the constant offset
`seated_diagnostics` had recorded but never resolved. An "occupied" sample is the
max over 3 antennas and ±1 bin at the GT cell of an in-FOV radar; "empty" is any
cell clearing every person by 1.5 m; the first 1500 frames of each window are
dropped as filter transient.

```
   win scenario             class             n        p50        p90        p99  perio/mag p50
    16 1 walking            walking       31522     198.88    1586.69    8218.58          1.811
    16 1 walking            empty        117987      13.33     193.16    1860.59          2.098
     5 4 walking            walking      120210     907.71    5424.13   29658.69          6.448
    19 1 walking            walking       30702     201.75    1476.13    9935.31          1.955
     4 1 walking + 1 seated walking       30712     304.45    3324.48   26391.30          2.775
     4 1 walking + 1 seated seated        36000     410.79    1616.62    2768.51          8.161
    10 2 walking + 2 seated walking       60722     457.74    4035.63   14770.32          4.157
    10 2 walking + 2 seated seated        66000     537.55    1997.64    4672.94          7.702
    21 1 seated             seated        36000      10.61     343.59     563.81          1.081
    21 1 seated             empty        116640       1.63      11.96      36.94          0.372
    22 empty room           empty        203040       1.19       5.87      25.73          0.286
    23 empty room           empty        203040       1.22       6.73      32.48          0.293
```

**A trap in this table, worth recording because it would have inverted the
verdict.** Pooled over all windows the comparison looks nearly fine — walking p50
542 against seated p50 354, only **1.53×**. That number is an artifact. The
"seated" cells in the *mixed* windows read 411 and 538 against w021's **10.61**,
because a range bin is a *shell* around a radar and a walker crossing that shell
dumps its much larger energy straight into the seated subject's own cell. The
only uncontaminated seated measurement in the dataset is w021, the seated-only
window — which is the window this entire investigation is about:

```
                                            p50        p90   vs w021 seated
  w021 seated (no walkers present)        10.61     343.59             1.0x
  w016 walking (no seated present)       198.88    1586.69            18.7x
  w005 walking (no seated present)       907.71    5424.13            85.5x
  w019 walking (no seated present)       201.75    1476.13            19.0x

  GATE 1  seated-occupied vs empty (both from w021)  =     6.51x   (need clear separation)
  GATE 2  walking-occupied vs seated-occupied, clean  =    57.61x   (need <= ~1x)
          same, pooled incl. contaminated mixed windows =   1.53x   (flattered)
```

**GATE 1 passes: 6.51×.** Static-occupied *is* separable from static-empty, and
the `perio/mag` ratio reproduces Step 7's finding almost exactly — 1.081 at
w021's seated cell against 0.372 at its empty cells, a **2.9× contrast**, the
same 2.9× `breath` delivered (0.489 / 0.173). Two different mechanisms, the same
static-occupied discrimination.

**GATE 2 fails, and fails worse than Step 7 did: 18.7–85.5×,** against `breath`'s
5–13×. This is the `|A|²` scaling showing up on real data — `breath` responded
∝|A| and `perio` responds ∝|A|², so walking's dominance is roughly *squared*
(13² ≈ 170, and the per-frame peak table reads w005/w021 = 213× for `perio`
against 12.8× for `breath`).

**3. The measurement that settles it, and it is new to Phase 4.** No previous
step asked what survives `normalize_channels`. The model sees `clip(x, 0,
p99)/p99` with **one** p99 pooled over every training window, so a statistic that
walking windows blow out leaves the seated subject crushed against zero however
well it separates in absolute terms:

```
  channel    pooled p99   w021 seated p50  as frac of p99  int8 levels   note
  mag            236.21              9.82          0.0416         10.6   deployed baseline
  breath          45.12              3.56          0.0790         20.1   Step 7: +1.2 to +2.1 SE seated
  perio         3607.53             10.61          0.0029          0.8   this channel
```

The input quantizer spans [0, 1] in 255 int8 steps, so "int8 levels" is how many
quantization steps the median seated cell rises above an empty one. **`breath`
delivered 20.1 levels and bought a measured +1.2 to +2.1 SE seated gain;
`perio` delivers 0.8** — 25× less than `breath` and **13× less than plain
magnitude**. Below ~1 level the seated subject is not representable in the
deployed int8 input at all. This is a *prediction* the gate can make and the CV
run would only have confirmed: the channel would have behaved as a walking
detector with no seated gain to show for it.

**A hard deployability finding, independent of accuracy.** A conjugate product at
lag L needs the actual sample from L frames ago, so every frame must be retained
for 100 frames and no recursion can compress that:

| channel | SRAM for all 846 cells | what it stores |
| --- | --- | --- |
| `breath` (Step 7) | **43.0 KB** | 6 recursive states per cell, any lookback |
| 4.1d ring buffer | 183.0 KB | 83 past *magnitude* frames |
| **`perio` (this)** | **680.8 KB** | 100 past *complex* frames — 2×/frame vs 4.1d |

**680.8 KB exceeds the ESP32-S3's 512 KB SRAM budget on its own**, before the
~104 KB the streaming declutter already uses. `evaluate_constraint.py` cannot see
this — it measures the TFLite arena, and the buffer lives in `code.py` outside
the graph. The recursive-filter advantage that made `breath` 4× cheaper than
4.1d does not transfer to a lagged product.

**Dtype, measured at both precisions because the guidance has now been wrong in
both directions.** Unlike `breath`, this filter is **stable in float32** — 2.1e−5
relative error over a full 7500-frame window, on both real seated and real
4-walker data. The reason is not pole placement, and this is the transferable
part: `perio`'s poles sit at radius **0.9967**, *closer* to the unit circle than
`breath`'s 0.9877. What kills `breath` in float32 is **direct-form tap
cancellation** — its denominator taps `[1, −5.76, 13.83, −17.75, 12.83, −4.95,
0.80]` sum to ~3e−9, a cancellation factor of **3.5e8**, far past float32's ~1e−7
resolution, while a one-pole EMA's `[1, −α]` cancels by only **599×**. So
`breath` needs an `sosfilt` biquad cascade to deploy and `perio` does not. The
verifier asserts float32 *stability* here, the mirror image of
`verify_breath_channel.py`'s divergence guard, so neither note can be copied onto
the wrong channel later.

#### What this closes, and what it opens

The periodicity premise is **not** falsified — this is the important distinction
from Steps 3–7. Coherence at the respiration lag separates breathers from walkers
essentially perfectly (1.0000 vs 0.0000–0.24) and the fixed-lag design works. What
is falsified is **this estimator of it**: an *unnormalized* conjugate product is
an energy statistic wearing a periodicity statistic's clothes, and Phase 4 has now
lost three candidates to the same root cause — `breath` (∝|A|), `perio` (∝|A|²),
and 4.1d — because **walking's amplitude advantage swamps any absolute-magnitude
feature, whatever temporal structure is built on top of it.**

The decomposition printed in the synthetic table says exactly where a successor
goes: divide by the smoothed `|z|²` and return the **coherence** itself, a
normalized correlation coefficient in [0, 1] that is amplitude-*invariant* by
construction. On the synthetic that statistic reads 1.0000 for every breather and
≤0.24 for every walker regardless of amplitude, which is precisely the ordering
both gates demand. It was not built here because it is a different mechanism, not
a constant change, and this project's rule is one variable per run — but it is now
the cheapest untested idea in Phase 4, and it inherits a finished harness that
will judge it in ~4 minutes.

**Caveats, stated plainly.**

1. **No CV run was spent, so there is no F1, no paired SE, and no walking-recall
   number for this channel.** The adoption bar was never reached; the gate is a
   *prediction* of the run's outcome, grounded in a 0.8-int8-level input signal
   and a reference channel with a known result. It is not a measurement of the
   run.
2. **The 20× walker amplitude in the synthetic is a stand-in**, taken from Step
   7's measured per-frame peaks (601–1652 walking vs 41 seated). The real-data
   gate does not depend on it.
3. **The serpentine synthetic re-crosses its cell every 4 s**, a designed worst
   case rather than a measured lap time. It is reported, not asserted — but note
   w019, the real serpentine walker, reads 19.0× a seated subject on real data,
   so the effect does not depend on the caricature.
4. **The `|A|²` scaling argument predicts the walking/seated ratio should be
   roughly the square of `breath`'s.** Measured 18.7–85.5× against 5–13×, and
   213× against 12.8× on per-frame peaks — consistent, but the two channels also
   differ in smoothing and in the max-across-lags reduction, so this is an
   explanation that fits rather than an isolated variable.

### Step 9 — 4.1f: the amplitude-invariant coherence. Every gate passed; the run was the largest regression in Phase 4

*2026-08-29, on Christian's machine. **Two CV runs** (a fresh 4-radar `mag`
baseline and the candidate). Negative, decisively, on every metric at every
matched threshold. Adopt nothing. This closes the periodicity/coherence line.*

**The hypothesis, and it was the right one to test.** Step 8 left exactly one
cheap idea standing and named it precisely: the smoothed conjugate product
decomposes as `|A|² × coherence`, the coherence factor already measured 1.0000
for every synthetic breather against ≤0.24 for every walker, so *divide by the
smoothed `|z|²` and return the coherence itself*. Amplitude-invariant by
construction: scale `z` by any `k` and numerator and denominator both scale by
`k²`. That is an algebraic identity, not a tuning choice, and it is the exact
antidote to the failure that had by then claimed three candidates in a row.

The mechanism did everything it was supposed to. The model still got worse at
everything, including seated subjects. **That gap — between a feature that
measures the right thing and a model that can use it — is the finding.**

#### The two fixes

**Fix 1, the normalization.** `_perio` now returns
`|smoothed(z(t)·conj(z(t−lag)))| / smoothed(|z(t)|²)`. One energy EMA, computed
once and shared by every lag (the energy level of a cell does not depend on
which lag is being tested), at the *longest* lag's time constant — a denominator
with a shorter memory than the numerator's lag could forget the energy it is
supposed to normalize and let the ratio blow up.

**Two cold-start hazards, both of which a ratio has and a raw product does not.**
The first is the obvious one: a cell that is exactly zero gives 0/0, guarded by
a tiny absolute floor (`PERIO_ENERGY_FLOOR = 1e-30`, ~30 orders below an empty
real cell, so it never touches a measured value — a floor anywhere near the
signal scale would manufacture a cold-start spike rather than prevent one).

The second was not anticipated by the design brief and is worth recording. Since
`_lagged` clamps to `z₀`, the frame-0 conjugate product is `z₀·conj(z₀) = |z₀|²`
— **the denominator exactly** — so a numerator EMA warm-started there makes
*every cell in the room* read coherence **1.0000 on frame 0**. A full-scale
phantom on the one frame inference cannot skip. The numerator EMA therefore
zero-starts while the energy EMA keeps the warm start (`|z₀|²` genuinely *is* an
energy observation), which bounds frame 0 at `1/(3·min lag)` = 0.0056, measured.
**This is invisible in the unnormalized channel**: `|z₀|² ≈ 1e−23` is nothing in
absolute units and only becomes a full-scale value once divided by itself.

It was caught because check 4 of the verifier was repaired — see below.

**Fix 2, the SRAM budget, and this one is not free.** 680.8 KB was measured on
the buffer *in isolation*. The real question is the combined total, so all three
costs were computed from the code that would incur them:

| component | bytes | KB |
| --- | --- | --- |
| streaming declutter working set | 103,680 | **101.2** |
|   — `zi` state (6×3×120×2, float64) | 34,560 | 33.8 |
|   — frame + background + output | 69,120 | 67.5 |
| model activation arena (measured, not quoted) | 28,262 | **27.6** |
| `perio` history + filter state (564 cells) | 381,264 | **372.3** |
|   — z history, 82 × (re,im) × 4 B | 369,984 | 361.3 |
|   — 2 numerator EMA states, complex | 9,024 | 8.8 |
|   — 1 shared energy EMA state, real | 2,256 | 2.2 |
| **TOTAL** | **513,206** | **501.2** |
| ESP32-S3 SRAM budget | 524,288 | 512.0 |
| **headroom** | **11,082** | **+10.8 (+2.1%)** |

The declutter figure is *derived* here rather than repeated — the four arrays
`remove_clutter_streaming` holds per frame — and it independently reproduces the
"~104 KB" the task board had recorded without a derivation (103,680 B is 104 kB
decimal). The arena is re-measured every run: the verifier builds the candidate
architecture through `quantize.py`'s own export path and shells out to
`evaluation/evaluate_constraint.py` for the number, because that script is the
authority and the figure is known to vary by machine.

**What each lever is worth**, so the choice is visible rather than asserted:

```
   radars lags            buffer KB  total KB  verdict
        6 (60, 82, 100)       684.1     812.9  over budget
        6 (60, 82)            558.5     687.3  over budget
        6 (82,)               551.9     680.7  over budget
        4 (60, 82, 100)       456.0     584.9  over budget
        4 (60, 82)            372.3     501.2  OK  <- this run
        4 (82,)               367.9     496.8  OK
```

**Both levers are needed and neither alone suffices.** Radars first, because the
geometric feasibility check had already established zero blind spots on all 11
seated identities for radars 1,2,4,6. Then lag 100, which the buffer is sized by
— and which is cheap to lose, since Step 8 measured a breather placed *between*
candidate lags keeping 1.000× the on-lag response. Keeping 60 alongside 82 costs
8 B/cell (4.4 KB), so it stays; dropping to a single lag would buy only 4.4 KB.
**10.8 KB of headroom is real but thin** — it covers the three costs this project
controls and not the IDF's own stack, heap fragmentation, or driver buffers.

#### The gate, repaired in four places, and all of it passed

**Check 4 was unreachable dead code.** An early `return` sat above the entire
output-contract and cold-start section while the verdict block printed
`contract, cold start, ... : PASS (asserted)`. It had never run. Repairing it is
what caught the frame-0 coherence-1.0 phantom above, in this branch's own first
implementation — the check was worth exactly what it was supposed to be worth,
the moment it was allowed to execute.

**Amplitude invariance is now asserted, not just printed.** Step 8's table had a
coherence column and no check on it. The same breather at 1× and 1000× amplitude
(a 10⁶× change in `|z|²`):

```
  cell                                    |z|^2  coherence
  breathing, lag 82, x1 amp            0.010000     1.0000
  breathing, lag 82, x1000 amp     10000.000000     1.0000

  energy ratio between the two cells      1000000.0x
  coherence ratio between the two cells     1.000000x  (relative drift 0.00e+00)
```

Exact invariance, to the last bit. The rest of the synthetic:

```
  cell                         kind          coherence        |z|^2  vs breather
  breathing, lag 82 (0.305 Hz) periodic         1.0000     0.010000        1.00x
  breathing, lag 70 (0.357 Hz) periodic         1.0000     0.010000        1.00x
  breathing, lag 91 (0.275 Hz) periodic         1.0000     0.010000        1.00x
  transient 1 s, x20 amp       transient        0.0000     0.017725        0.00x
  transient 3 s, x20 amp       transient        0.0255     0.053174        0.03x
  transient 3 s, x1 amp        transient        0.0255     0.000133        0.03x
  gait 2 Hz burst, x20 amp     adversarial      0.0913     0.070898        0.09x
  serpentine, re-crosses/4 s   adversarial      0.4451     0.695602        0.45x
  empty (flat zero)            empty            0.0000     0.000000        0.00x
  empty (noise)                empty            0.0687     0.000002        0.07x
```

Every walker now scores below every breather — including the 20×-amplitude ones,
which is the comparison the uncorrected channel lost. Note the two rows that
matter later: `transient 3 s` reads **0.0255 at both 1× and 20× amplitude**
(invariance working as designed), and **`empty (noise)` reads 0.0687** — not
zero. Hold that number.

**GATE 3 was promoted to a hard gate and its arithmetic corrected.** Two fixes:
the real `clip(x, 0, p99)/p99` is applied (this channel's pooled p99 lands
*below* its own seated value, so the seated cell saturates and an unclipped
fraction would have overstated it by 2×), and the contrast is measured *above an
empty cell* rather than as an absolute fraction of p99. For `mag` those agree to
within a level; for a channel with a non-zero empty floor they do not.

**And every clause passed, several of them by a wide margin.**

```
  win scenario             class             n        p50        p90        p99  perio/mag p50
    16 1 walking            walking       21558     0.0738     0.1197     0.1753        0.00065
    16 1 walking            empty         79032     0.0480     0.0890     0.1539        0.00762
     5 4 walking            walking       81678     0.0807     0.1133     0.1531        0.00053
    19 1 walking            walking       21129     0.0803     0.1209     0.2005        0.00074
     4 1 walking + 1 seated seated        24000     0.0900     0.1315     0.1672        0.00180
    10 2 walking + 2 seated seated        42000     0.0851     0.1170     0.1459        0.00112
    21 1 seated             seated        24000     0.3353     0.7366     0.8127        0.02915
    21 1 seated             empty         77760     0.0570     0.0943     0.2642        0.01305
    22 empty room           empty        135360     0.0548     0.0862     0.1170        0.01328
    23 empty room           empty        135360     0.0543     0.0862     0.1195        0.01314

                                            p50        p90   vs w021 seated
  w021 seated (no walkers present)       0.3353     0.7366            1.00x
  w016 walking (no seated present)       0.0738     0.1197            0.22x
  w005 walking (no seated present)       0.0807     0.1133            0.24x
  w019 walking (no seated present)       0.0803     0.1209            0.24x

  GATE 1  seated-occupied vs empty (both from w021)  =     5.88x   (need > 2x)
  GATE 2  walking-occupied vs seated-occupied, clean  =     0.24x   (need <= 1.5x)
          same, pooled incl. contaminated mixed windows =   0.84x   (flattered)

  channel   pooled p99  w021 seated  w021 empty  norm seated  norm empty  levels vs empty
  mag         246.5985      11.5020      4.3667       0.0466      0.0177              7.4
  breath       44.7893       6.1776      0.9969       0.1379      0.0223             29.5
  perio         0.1884       0.3353      0.0570       1.0000      0.3024            177.9

  GATE 1 PASS   GATE 2 PASS   GATE 3 PASS
  SRAM GATE PASS  -- 501.2 KB < 512 KB
```

**GATE 2 at 0.24× is the number Phase 4 had been chasing for four steps.**
Walking reads *below* a seated subject, on real data, for the first time — where
`breath` was 5–13× and uncorrected `perio` 18.7–85.5×. And GATE 3's 177.9 int8
levels is 24× `mag`'s 7.4 and 6× `breath`'s 29.5, where `breath`'s 29.5 had
bought a real +1.2 to +2.1 SE seated gain. On the evidence available before the
run, this was the strongest-looking candidate in Phase 4 by a wide margin.

#### The CV runs, and they are unambiguous

**Two runs, because the SRAM fix forces 4 radars and comparing a 4-radar
candidate against the 6-radar baseline would confound the channel with the radar
count** — precisely the criticism 4.1d earned. The paired reference is therefore
a **freshly retrained same-machine 4-radar `mag` baseline** (discipline #10), so
the only variable between A and B is the channel. `mag+perio`'s argmax moved to
0.20 (baseline 0.30), so the whole matched range is reported (discipline #7).

Paired jackknife, `mag+perio` (A) vs `mag` (B), both on radars 1,2,4,6:

| matched th | pooled F1 A / B | Δ (paired SE) | walking recall Δ | seated recall Δ |
| --- | --- | --- | --- | --- |
| 0.40 | 0.5852 / 0.8080 | **−0.2228 (−10.3 SE)** | **−0.3087 (−9.5 SE)** | **−0.3483 (−5.3 SE)** |
| 0.30 | 0.6688 / 0.8085 | **−0.1397 (−10.9 SE)** | **−0.1966 (−9.6 SE)** | **−0.3113 (−5.2 SE)** |
| 0.20 | 0.6826 / 0.7890 | **−0.1064 (−8.1 SE)** | **−0.1117 (−11.0 SE)** | **−0.2404 (−4.6 SE)** |
| 0.10 | 0.6408 / 0.7543 | **−0.1136 (−9.0 SE)** | **−0.0734 (−12.5 SE)** | **−0.1644 (−3.5 SE)** |

**WALKING RECALL: FAIL.** −9.5 to −12.5 paired SE at every matched threshold.
Worse than `breath`'s −5.3 to −10.4, which was already the clause that killed it.

**SEATED RECALL: FAIL, and in the wrong direction.** −3.5 to −5.3 SE. This is
the part that is not a repeat of anything: every prior Phase 4 candidate either
moved seated recall up or left it flat. This one made the seated subject
*harder* to find, using a channel measured at 177.9 int8 levels of seated
contrast. w021 specifically went **0.7% → 0.1%** at a matched 0.30 (1.3% at its
own argmax 0.20, against the baseline's 0.7%).

**Empty-room phantoms: fewer, and it buys nothing.**

| matched th | `mag+perio` | `mag` (4r) | ratio |
| --- | --- | --- | --- |
| 0.40 | 134 | 1,962 | ÷14.6 |
| 0.30 | 930 | 7,571 | ÷8.1 |
| 0.20 | 3,502 | 13,972 | ÷4.0 |
| 0.10 | 14,897 | 15,159 | ÷1.0 |

The same single axis Phase 4 has now watched seven interventions slide along —
how eagerly the model fires on a static scene — and this one slid down it hard
enough to lose the detections too.

**STRICTLY DOMINATED BY THE THRESHOLD KNOB (discipline #12), more completely
than 4.1b was.** The baseline at th=0.40 beats `mag+perio` at its own best
th=0.20 on **all three** metrics simultaneously: seated recall 0.5404 vs 0.4423,
phantoms 1,962 vs 3,502, pooled F1 0.8080 vs 0.6826. Higher recall, fewer
phantoms, higher F1, no retraining. There is no operating point at which this
channel buys anything free post-processing could not.

#### Why it failed, and the number was on the gate's own screen

Read the GATE 3 table again, the `norm empty` column:

| channel | empty cell, after `clip/p99` | vs `mag` |
| --- | --- | --- |
| `mag` | 0.0177 | 1.0× |
| `breath` | 0.0223 | 1.3× |
| **`perio`** | **0.3024** | **17.1×** |

**Amplitude invariance is exactly what removes the channel's ability to say
"nobody is here."** A cell containing nothing still contains noise, and
uncorrelated noise smoothed over `N_eff ≈ 3·lag` samples self-correlates at
~`1/√N_eff` ≈ 0.06 — visible in the synthetic as `empty (noise) 0.0687` and on
real data as an empty-cell floor of 0.054–0.057 that is *identical* in empty
rooms, walking windows and w021. Dividing by energy normalizes that floor up to
**30% of full scale**. So the model receives a second input plane whose
background sits 17× higher than the magnitude channel's and carries substantial
variance, across all 845 cells where nobody is. The early-fusion `Dense` flattens
every cell of every channel into one vector, so this is not a small perturbation:
it is half the input dimensionality turned into a high-pedestal nuisance plane.

The occupied-vs-empty *contrast* was excellent (5.88×, 177.9 levels). The
*pedestal* was catastrophic. **The gate measured the first and printed the
second.** GATE 3's threshold was applied to the contrast column while the
disqualifying number sat two columns to its left, unexamined.

This is discipline #9 — "a per-cell d′ is a feasibility check, not a prediction;
the model sees 846 cells and the other 845 carry the noise" — reappearing in the
one case where the per-cell measurement was not marginal but spectacular. A
27-fold better per-cell contrast than the deployed baseline still produced the
largest regression in Phase 4.

A second, independent mechanism is visible in the gate table and points the same
way: **in mixed windows the coherence of a seated subject collapses.** w004 reads
0.0900 and w010 0.0851, against w021's 0.3353 and an empty floor of ~0.05. A
walker crossing the seated subject's range shell injects incoherent energy that
raises the *denominator* and destroys the ratio. Step 8 recorded contamination
*inflating* a seated cell ~40×; under normalization the same physics *suppresses*
it. Since **7 of the 8 seated-containing windows are mixed**, the channel was
only ever going to carry real information in w021 — 1 window in 24 — while
carrying the nuisance pedestal in all of them. This was visible before the run
and was flagged as a caveat rather than as a stop; it should have been a stop.

#### A correction to the 4.1d record, obtained free

The 4-radar `mag` baseline is the separating control 4.1d itself said was needed
and never ran ("the separating control, if ever wanted, is `--radars 1,2,4,6
--channels mag`"). Now it exists, so 4.1d can be rescored against it:

| comparison | pooled F1 Δ | walking recall Δ | seated recall Δ |
| --- | --- | --- | --- |
| 4.1d vs **6-radar** `mag` (as recorded) | −0.0839 (**−9.1 SE**) | — | −1.6 to −1.9 SE |
| 4.1d vs **4-radar** `mag` (this control, th 0.30) | **−0.0038 (−0.5 SE)** | −0.0221 (−6.4 SE) | **−0.0063 (−0.3 SE)** |

**Almost all of "the largest regression in Phase 4" was the radar reduction, not
4.1d's channels.** Against a matched-radar control its pooled F1 is −0.5 SE
(noise) and its seated recall −0.3 SE (noise); only walking recall is a real
−6.4 SE effect. The verdict on 4.1d does not change — it is still not adoptable
and its channels still bought nothing — but the *reason* recorded for it was
wrong, and the confound its own spec flagged turns out to have carried the
effect. Recorded here rather than edited into Step 5, per this file's convention.

**And the radar reduction is expensive**, which is a Phase 5 constraint in its
own right:

| comparison | pooled F1 | walking recall | seated recall |
| --- | --- | --- | --- |
| 4-radar `mag` vs 6-radar `mag`, matched 0.40 | −0.0688 (**−10.7 SE**) | −0.0868 (−8.5 SE) | −0.0622 (−2.0 SE) |
| same, matched 0.30 | −0.0628 (**−11.1 SE**) | −0.0547 (−8.0 SE) | −0.0468 (−2.2 SE) |

Dropping two radars costs ~0.065 pooled F1 — comparable to the *entire* seated
ceiling (+0.057 to +0.062). The geometric check that found zero seated blind
spots was about *coverage*, and coverage is not accuracy: the redundancy the two
extra radars provide is worth a great deal even where it is not needed to see a
subject at all. **Any future feature whose SRAM cost forces 4 radars starts
0.065 F1 in the hole and has to pay that back before it breaks even.**

**Caveats, stated plainly.**

1. **The two CV runs isolate the channel and nothing else** — same machine, same
   code, same seeds, same 4 radars, `--channels mag` vs `--channels mag+perio`.
   The radar-count comparison in the last table is a *separate* pairing against
   an earlier same-machine 6-radar run, and is not confounded with the channel.
2. **The `mag+perio` model has 403,043 params against the baseline's 258,659**
   (both verified), because a second channel widens the early-fusion `Dense`. So
   A also has 1.56× B's capacity, and it still lost by 8–11 SE. That direction of
   confound makes the negative result stronger, not weaker.
3. **The pedestal explanation is an explanation that fits, not an isolated
   variable.** It is supported by three independent measurements (the synthetic
   noise cell at 0.0687, the real empty floor at 0.054–0.057 identical across
   scenario types, and the 0.3024 normalized empty value) and it predicts the
   observed harm to *walking* recall, which no seated-subject mechanism explains.
   But no run was spent isolating it, and none should be.
4. **`perio`'s pooled p99 came out 0.18146 in the actual training run against the
   gate's 0.1884 estimate** — 4% apart, which confirms the gate's p99
   methodology was sound. The gate's failure was in which column it gated on,
   not in its arithmetic.
5. **The serpentine synthetic reads 0.4451**, the highest non-breather in the
   table, and lag 100 having been dropped for SRAM happens to help it (the
   caricature re-crosses every 100 frames). That is a coincidence of the budget,
   not a design property, and is not claimed as one.

#### What this closes

**The periodicity/coherence line is finished.** Not "unpromising" — measured, on
a run whose pre-run gate it passed on every clause, and lost by 8–11 SE. The
premise survives intact for a third time (coherence *does* separate breathers
from walkers, 1.0000 vs ≤0.24, and on real data 0.24× is the best class ordering
Phase 4 has produced). What is now falsified is the whole family: **both forms of
the periodicity statistic have been built and both failed, for opposite reasons.**
The unnormalized form is an energy statistic that walking wins; the normalized
form has no concept of an empty cell and floods the input with a nuisance
pedestal. There is no third form — normalizing is a binary choice, and each
branch has now been measured.

Per this project's standing rule, a tenth variant needs a mechanism the nine
before it do not already falsify. Between Step 8 and Step 9 that bar is higher
than it has ever been, because the two steps close opposite halves of the same
space. **This is a stopping point, and it is recorded as one.**

### Step 10 — 4.3: the decode, not the input. The first positive on the seated mode

Steps 3-9 all asked the same question: how do we make a motionless body visible
to a model that sees one 40 ms frame at a time? Nine input representations later
the answer was still "we cannot". Step 10 asks a different question — **is the
model already seeing it, and is the decode throwing it away?** — and the answer
is yes, for three quarters of the seated misses.

#### The diagnosis, which splits one error mode into two

Measured on the pooled OOF heatmaps of a freshly retrained `mag`/EMA baseline
(`logs/20260829-134433`, pooled F1 0.8803 @ 0.40, reproducing the recorded
0.8768 well inside 1 SE). Two statistics, per seated person-frame: whether the
deployed decode recalled it, and where the seated person's own grid cell ranked
among all 216 cells of that frame's heatmap.

| condition | seated recall | rank of the seated cell (216 cells) |
| --- | --- | --- |
| a walker within 1.0 m | 94.6% | — (contaminated: the walker's own detection matches within 1.0 m) |
| walker 1.0-1.5 m | 68.7% | — |
| walker 1.5-2.0 m | 64.6% | median **15** |
| walker 2.0-3.0 m | 64.2% | (chance would be 108) |
| walker 3.0-9.0 m | 60.0% | |
| **no walker in the room (w021)** | **0.6%** | median **203** — bottom 6% |

Two different failures had been aggregated into one number all along:

1. **Mixed windows.** The model puts real evidence at the seated cell — median
   rank 15 of 216, far above chance — and the per-frame threshold, which only
   ever sees the top few cells, discards it. **76% of all seated misses.**
   Recoverable without touching the model.
2. **w021, the seated-only window.** The cell ranks *below 94% of the room*.
   That is the 1.05x input measurement showing through: under an EMA background
   a seated-only window is indistinguishable from an empty one, so there is
   nothing in the heatmap to decode. **24% of seated misses, and no output-side
   method can reach it.**

Note the discontinuity between "walker 3-9 m away, 60.0%" and "no walker, 0.6%".
The walker is far in both cases, so this is not proximity contamination — the
mere presence of a moving body in the room is what makes a seated one findable.

#### The rule

Smooth the 18x12 output heatmap with a causal one-pole EMA (`lfilter`, seeded
from frame 0 — the `remove_clutter` idiom, so it streams), then decode
`max(P, S * threshold / t_lo)` at the unchanged 0.40 threshold. Equivalently: a
cell fires if it is bright now **or** has averaged above `t_lo` for a while. The
rescale keeps the result one surface, so NMS and the centre-of-mass refinement in
`extract_peaks_from_grid` still see a coherent peak.

Cost: one float32 array of the grid's shape — **864 bytes** — and ~430 flops per
frame. No retraining, no flash, no model change, no new input channel.

#### The result

Paired jackknife over the 24 windows, tau = 10 s, t_lo = 0.30, against the same
baseline:

| metric | persistence | baseline | delta | paired SE | SEs |
| --- | --- | --- | --- | --- | --- |
| pooled F1 | 0.8862 | 0.8803 | +0.0059 | 0.0048 | **+1.2** |
| seated recall | 0.6807 | 0.6214 | +0.0594 | 0.0161 | **+3.7** |
| walking recall | 0.9475 | 0.9470 | +0.0005 | — | flat |
| empty-room phantoms | 1893 | 1893 | 0 | — | bit-identical |

**+3.7 SE on seated recall is roughly double anything Phase 4 produced** (the
best was `breath` at +1.2 to +2.1 SE), and unlike `breath` it costs no walking
recall and no phantoms. At t_lo = 0.25 seated recall reaches 0.7105 (+4.8 SE)
with pooled F1 still +0.4 SE.

**Not dominated by the threshold knob (discipline #12).** The baseline buys
seated recall by lowering its threshold, so the comparison must be at matched
seated recall:

| at seated recall ~0.71 | pooled F1 | empty-room phantoms |
| --- | --- | --- |
| persistence tau=10 t_lo=0.25 | **0.8832** | **2,240** |
| baseline th=0.25 (seated 0.7155) | 0.8675 | 10,055 |

**+0.0157 F1 and 4.5x fewer phantoms for the same seated recall.** Free
post-processing cannot reach this operating point.

#### It is a trade, and the done gate caught it

| subset | F1 baseline -> persistence | false positives |
| --- | --- | --- |
| seated-containing (8 windows) | 0.8284 -> **0.8467** (+0.0184) | +1,893 |
| walking-only (14 windows) | 0.9219 -> **0.9193** (-0.0026) | +1,384 |
| empty (2 windows) | unchanged | +0 |

Every walking-only window is flat or negative. The pooled gain survives only
because 8 of 24 windows contain a seated subject, so **adopting this is a bet
that the test session's scenario mix resembles the training one.**

The example set makes the cost concrete: **F1 0.9884 -> 0.9768**, precision
0.9606, recall 0.9935. That set is 2 subjects, both walking, no seated subject
at all — neither stays within 0.5 m of their starting point for more than 11% of
frames — on data the model trained on, where it already scored 0.9884.
Persistence has nothing to gain there and only adds false positives. The OOF
number is the honest one (discipline #1), but the regression is real and the
gate is right to flag it.

#### The obvious repair, tried and failed

An EMA leaves a decaying **trail** behind a moving person, and the gain promotes
cells a walker has just vacated: w012 (3 walkers) gains +1,018 false positives
while the single-walker windows 016-020 gain exactly **0**. The hypothesis was
that a trail cell swings bright->dark while a seated cell holds steady, so gating
promotion on a low EMA of `|P - S|` should keep one and drop the other.

| steadiness gate k | pooled F1 | seated F1 | walking-only F1 | seated recall |
| --- | --- | --- | --- | --- |
| none (shipped) | 0.8862 | 0.8467 | 0.9193 | 0.6807 |
| 0.70 | 0.8866 | 0.8472 | 0.9197 | 0.6780 |
| 0.50 | 0.8846 | 0.8394 | 0.9217 | 0.6503 |
| 0.35 | 0.8811 | 0.8305 | 0.9219 | 0.6264 |
| 0.25 | 0.8804 | 0.8286 | **0.9219** | 0.6220 |

Tightening k restores walking-only F1 exactly as fast as it destroys the seated
gain — one axis again, the same shape six Phase 4 interventions produced. **The
seated evidence is itself variable, so the walking-only cost is intrinsic to the
mechanism rather than an implementation defect.** Shipped ungated: k=0.7's
+0.0004 pooled is noise and does not justify a second knob and a second state
array.

#### Caveats on record

- **tau and t_lo were selected on the same 24 windows they are scored on.** A
  9-point grid, and the surface is smooth and monotone in both knobs, but this is
  precisely the hazard Phase 2's structure exists to avoid. The magnitudes are
  optimistically biased. +3.7 SE on seated recall has room to survive that;
  **+1.2 SE on pooled F1 does not.**
- w021 is unchanged at 0.6% and cannot be fixed from the output side.
- The empty-room phantom count is bit-identical, so the one scenario the spec
  calls out explicitly is untouched.

#### Why this worked where nine input representations did not

Every earlier candidate tried to give a per-frame model better evidence. This one
uses evidence the model was already producing. A per-frame architecture cannot
represent "this cell has been warm for ten seconds" whatever it is fed — that
proposition only exists across frames, which is where the decode lives. The nine
negatives were not wasted: they are what establishes that the input side has been
searched thoroughly enough for the answer to be somewhere else.

### Geometric feasibility check for a 4-radar reduction (2026-08-27)

Room has 6 radars: 1 on each short wall (positions (2.4,0) and (2.4,7.2)),
2 on each long wall. Checked whether dropping to 4 radars (both short-wall
ones + one from each long-wall pair) creates a blind spot for any real seated
position: all 11 seated identities across the 8 seated windows, against each
radar's actual ±60° FOV. **Zero blind spots, in all 4 possible pairings.**
This is what makes the radar-count reduction viable as a way to buy budget for
more input channels without losing coverage on the population that matters.

### Built and kept (all merged, all reusable)

| artifact | what it does |
| --- | --- |
| `src/helpers/seated_diagnostics.py` | locates the signal with no trained model, empty rooms as the null |
| `src/helpers/oof_breakdown.py` | per-window OOF F1, seated subset scored apart, **per-person walking/seated recall**, and `--compare` for a paired A/B with jackknife SEs |
| `preprocessing.CHANNEL_SETS` | swappable input channels; adding one is a dict entry |
| `build_channels(..., radar_indices)` + `--radars` | swappable radar subset, threaded through training, OOF scoring and quantization, recorded to `src/models/best_radars.txt` |
| complex-CIR cache (`.cplx.npz`) | p99-, channel- AND radar-independent, so an ablation costs no re-declutter (9 s to rebuild all 24 on Christian's machine) |
| `logs/phase4-ablation/` | four run logs, per-window counts, paired-jackknife script — **on Mirko's machine only**; the paired jackknife was reimplemented as `oof_breakdown.py --compare` because of it |
| `logs/20260827-173410/` | 4.1d's OOF cache and its model artifacts, kept together so the run is reproducible |
| `--seated-oversample K` + `src/models/best_seated_oversample.txt` | per-fold training-set oversampling of the seated windows, K=1 a verified bit-exact no-op |
| `data_loader.load_seated_windows` | the 8 seated windows derived from `metadata.jsonl` instead of a second hardcoded list |
| `src/helpers/verify_oversample.py` | frame-level proof that no held-out window enters its own fold's training set, plus exact-K multiplicity and p99 invariance |
| `logs/20260828-120148/` (K=3) and `logs/20260828-125401/` (fresh `mag` baseline) | Step 6's paired OOF caches, on Christian's machine |
| `preprocessing.channel(z, "breath")` + `mag+breath` | causal 0.15–0.6 Hz respiration envelope of the complex CIR; one finished per-frame number, `zi`-initialized so frame 0 needs no clamp |
| `src/helpers/verify_breath_channel.py` | six asserted checks on the channel *before* it costs a CV run: passband, synthetic band separation, output contract, cold start, SRAM cost, and a float32-divergence regression guard |
| `logs/20260828-160547/` (`mag+breath`) and `logs/20260828-163339/` (fresh `mag` baseline) | Step 7's paired OOF caches; the baseline is byte-identical to `20260828-125401` on all 24 windows |
| `preprocessing.channel(z, "perio")` + `mag+perio` | periodicity at the respiration lags as a **normalized coherence** in ~[0,1] (Step 9; was the raw `\|A\|² × coherence` product in Step 8). **Not adoptable in either form** — Step 8 killed the unnormalized one pre-run, Step 9 measured the normalized one at −8 to −11 SE |
| `src/helpers/verify_perio_channel.py` | the pre-CV gate disciplines #13/#15/#16 ask for, and the most reusable artifact of Steps 8–9: a data-calibrated GT-position → radar-cell mapping (+6 bins, +0.90 m, two independent estimators), per-class real-data distributions at those cells restricted to the deployed radar subset, the **p99-survival / int8-levels** check, an **amplitude-invariance** assertion, a **combined-SRAM gate** (declutter + live-measured arena + channel buffer vs 512 KB), and a float32-stability guard. Exits 1 on failure. **Read Step 9's post-mortem before trusting it on a new channel: it passed a candidate that then lost by 8–11 SE, because it gated on the occupied-vs-empty contrast and merely printed the empty-cell pedestal** |
| `logs/20260829-010110/` (4-radar `mag`) and `logs/20260829-013631/` (4-radar `mag+perio`) | Step 9's paired OOF caches. The first is also the **separating control 4.1d never ran**, and it rewrites 4.1d's attribution |

### Lessons that generalize (also folded into the Measurement discipline
appendix as #6–#19, as live rules rather than just history)

1. **A per-cell d′ is a feasibility check, not a prediction.** It was real,
   reproducible, and the feature still failed in the full model — the network
   sees 846 cells and the other 845 carry the noise.
2. **An A/B on the same CV folds is paired; use the paired SE, not the
   single-model jackknife SE.** Per-window difficulty cancels — measured
   paired SE (±0.0089 pooled) is roughly half the unpaired figure (±0.018).
   This is what upgraded `mag+aoa` from "looks like noise" to "confirmed
   regression."
3. **Score the subset the change actually targets, at the identity level if
   relevant.** Pooled F1 hid the empty-room phantom effect entirely (a 7.4×
   cut is invisible in a metric where empty windows produce no true
   positives). The seated-*subset* F1 would have hidden the per-person
   collapse in w004 too. Only per-person recall caught it.
4. **Compare at a matched operating point, or the comparison is free to lie.**
   4.1d's threshold argmax moved 0.40 → 0.30. Read at each model's own argmax,
   its aggregate seated recall looks like a *gain* (62.2% vs the baseline's
   60.3%); at a matched threshold the same pair of runs is a 1.6–1.9 SE
   *loss*. The entire apparent improvement was the lower threshold buying
   recall for everyone. Quote both thresholds, and never compare a tuned
   operating point against an untuned one.
5. **A negative result gets cheaper the more it rules out.** 4.1d was run
   knowing four predecessors had failed, precisely because it was the last
   input-representation hypothesis with a real mechanism behind it. It cost
   one CV run to close an entire branch of the search space. That is a good
   trade, and it is the reason to write the negative results down in enough
   detail that nobody re-opens the branch.
6. **Regenerate the baseline; do not reuse a recorded number.** Step 6's
   baseline was retrained on the same machine with the same code immediately
   before the comparison, which cost ~30 min. It reproduced the recorded 0.8768
   and 73.2% exactly, and its OOF caches came out byte-identical to the
   pre-refactor run — so the retrain "wasted" half an hour and bought a proof
   that three merged infrastructure changes are bit-exact no-ops on the deployed
   path. When the adoption bar is ~1 SE on an unstable metric, the baseline is
   not the place to save time.
7. **Before trusting a data-side experiment, assert the split at frame
   granularity.** A grouped-CV leak does not throw, does not warn, and does not
   look wrong in any run output — it just inflates the score for the wrong
   reason. The check has to run against the real fold construction, not a
   re-implementation of it, or it proves only that two copies of the same bug
   agree.
8. **Ask whether an intervention beats the threshold knob before asking whether
   it beats the baseline.** Step 6's oversampled model was strictly dominated —
   worse seated recall, more phantoms, lower F1 — by the *baseline at a lower
   threshold*, a change that costs no retraining at all. A cross-threshold
   dominance check is not a substitute for a matched comparison, but it answers
   a different and equally decisive question: did this buy anything the cheap
   knob could not.
9. **A synthetic test validates the implementation, not the premise.** Step 7's
   `breath` channel passed every pre-CV check — the passband landed within
   0.01 Hz, a 0.3 Hz cell read 87× an equal-amplitude 2 Hz cell, and the test
   caught a real float32 NaN divergence that would have silently corrupted the
   channel 150 s into every window. It still could not predict the failure,
   because the failure was in the *modelling assumption* — real walking is a
   broadband 1–3 s transient with large in-band energy, not the out-of-band
   sinusoid the synthetic stood in for. **Pair every synthetic check with one
   histogram of the finished feature over real data of each class**; it costs
   seconds and it is the check that would have caught this before two CV runs.
10. **Verify a filter's numerics at the dtype you will actually run it at.**
    This bandpass has poles at radius 0.9877 and diverges to NaN at frame 3743
    in float32 direct form, while being perfectly stable in float64 and in a
    float32 biquad cascade. None of that is visible from the frequency
    response, and a NaN 150 s into a 300 s window would not have looked like a
    filter bug downstream. **Refined by Step 8: the criterion is not pole
    radius but direct-form tap cancellation.** `perio`'s poles sit at 0.9967 —
    *closer* to the unit circle — and it is perfectly stable in float32,
    because a one-pole EMA's taps cancel by 599× against the bandpass's 3.5e8×,
    and float32 resolves ~1e-7.
11. **The pre-run gate pays for itself the first time it fires, and it should
    exit non-zero.** Step 8 cost ~4 minutes of measurement and zero CV runs,
    against Step 7's two runs (~90 min) for a comparable verdict. The gate has
    to be a *program that fails*, not a section of printed numbers a reader is
    trusted to interpret — the numbers Step 7 needed were computable before its
    first run and nobody computed them.
12. **Score the feature AFTER normalization, not before.** A channel is fed to
    the model as `clip(x, 0, p99)/p99` with one p99 pooled over all training
    windows, so a statistic that walking windows blow out arrives at the seated
    subject as ~0 however well it separates in absolute terms. Measured in int8
    levels: `mag` 10.6, `breath` 20.1, `perio` 0.8. `breath` bought a real
    +1.2–2.1 SE seated gain on 20.1 levels; 0.8 levels cannot buy anything.
    **Every prior Phase 4 step measured d′ or raw separation on the unnormalized
    channel** — this is the cheap check none of them ran.
13. **Never pool a per-class statistic across mixed-class windows without
    checking for cross-contamination.** Step 8's pooled walking/seated ratio was
    1.53× and its clean one was 57.6×, because a range bin is a *shell* around a
    radar: a walker crossing that shell dumps its energy into a seated subject's
    own cell, inflating the "seated" number by ~40× in mixed windows. The pooled
    figure would have passed the gate and sent a doomed channel to a CV run.
    **Step 9 adds the mirror case:** under a *normalized* statistic the same
    physics *suppresses* instead of inflating — a walker's incoherent energy
    raises the denominator, collapsing a mixed-window seated subject from 0.335
    to ~0.087. The direction of the contamination flips with the statistic; the
    need to use single-class windows as the reference does not.
14. **A contrast at the cells you care about is not the whole input. Measure the
    PEDESTAL over the cells you don't.** Step 9's channel scored 177.9 int8
    levels of seated contrast against `mag`'s 7.4 — a 24× better per-cell
    feature — and produced the largest regression in Phase 4 (−8 to −11 SE on
    everything, seated recall included). The reason was in the same printed
    table, one column to the left: its empty cells normalize to **0.302 of full
    scale against `mag`'s 0.018**, because an amplitude-invariant statistic has
    no way to express "nobody is here" and uncorrelated noise still
    self-correlates at ~1/√N_eff. 845 of 846 cells are empty at any instant, and
    the early-fusion `Dense` sees all of them. **Gate on both columns:** a
    candidate needs contrast at the target AND a background no worse than the
    channel it is being added to.
15. **Normalizing a feature is a binary choice, so measuring both branches closes
    the family.** `perio` unnormalized is an energy statistic walking wins (Step
    8); `perio` normalized is a pedestal generator (Step 9). Neither is fixable
    by a third form, because there is no third form. When two candidates fail for
    *opposite* reasons, that is stronger evidence of a closed branch than two
    failing for the same reason — and it is the point at which to stop rather
    than to iterate.

### Cost

11 × ~20–45 min CV runs (4 on Mirko's machine, 7 on Christian's), one
complex-CIR cache rebuild (9 s), ~8.4 hours wall clock total across the four
branches — **plus Step 8 at ~4 minutes and zero CV runs**, which is the whole
argument for the pre-run gate, and **Step 9 at two runs that the same gate
cleared and should not have**, which is the argument for gating on the right
column.

---

## Edge optimization (Phase 5, `feat/conditional-inference`)

### Step 11 — 5.1: conditional inference, and the gate that could not be made safe

**The idea.** A per-frame model spends the same compute on an empty room as on
four walking people. After decluttering, an empty room is nearly silent, so one
scalar per frame decides whether to invoke the interpreter at all. Below a
calibrated gate the model is skipped and the all-zero heatmap it would have
produced is substituted; `Persistence` and `Tracker` still consume that frame, so
their state stays causal and a skipped frame is indistinguishable downstream from
a frame the model saw nothing in.

Cost of the gate itself: a mean over the 846 normalized cells, against ~330k
MACs for the model. **A skipped frame costs ~0.3% of an inference.**

This lives entirely in `submission/code.py`, outside the graph. That matters —
see Step 13 on why the in-graph version cannot ship.

**Choosing the statistic, and the way the obvious answer was wrong.** Two
candidates, both reduced from the normalized `mag` channel:

| statistic | occupied mean | empty mean | pooled d′ | empty frames skipped at its own safe gate |
| --- | --- | --- | --- | --- |
| `mean` | 0.08683 | 0.01651 | 2.765 | **95.1%** |
| `max` | 0.92892 | 0.11703 | **5.537** | 77.0% |

`max` separates the two classes twice as well by d′ and is the **worse gate**.
Its d′ is inflated by walking frames saturating at 1.0 — the median occupied
`max` is exactly 1.00000 in 21 of 22 occupied windows — which is a statement
about how bright a walker is, not about the empty-versus-quiet boundary the gate
actually sits on. Measurement discipline #9 again, in a new costume: a summary
statistic measured far from the decision boundary does not predict behaviour at
it. `mean` ships.

**The gate cannot be calibrated on "never skip an occupied frame", and this is
the finding.** The first calibration attempt asked for exactly that, with a 0.1%
budget, and reported a violation in **every single window**. Two separate causes,
and separating them was the whole work:

1. **Frame 0, in all 24 windows.** `remove_clutter` seeds its EMA from frame 0,
   so `decluttered[0]` is identically zero — measured at ~1e-15 after
   normalization, against ~0.08 for a normal occupied frame. It is a cold-start
   artifact, not a measurement, and the model has nothing to see there either.
   It is excluded from every calibration statistic and it is always gated. (This
   is the same fact `channels.lagged` documents from the other side: the clamp
   degrades gracefully because z₀ is ~0.)

2. **Window 021, and this one is real.** A single seated subject, nobody walking.
   Its occupied frames are **dimmer than an empty room**:

   | window | kind | occupied median | occupied p01 | occupied min |
   | --- | --- | --- | --- | --- |
   | w021 | seated only | **0.01693** | 0.01486 | 0.01388 |
   | w022 | empty | 0.01627 | — | — |
   | w023 | empty | 0.01632 | — | — |

   A ratio of **1.04**. This is the same 1.05× the Phase 4 diagnostics measured
   on the input side, arriving now as a hard limit on a completely different
   mechanism. Honouring "never skip an occupied frame" for w021 forces the gate
   below 0.01388, which skips **0.06% of empty-room frames** — the feature buys
   nothing at all.

**The rule that was adopted instead.** Windows are split by whether a person is
separable from an empty room *at all*: this window's 1st-percentile occupied
energy must exceed the pooled empty-room median. w021 fails that test — by
measurement, not by name, which is what makes the rule portable to a dataset
where a different window is the pathological one. The gate is then set to a 0.9
margin below the dimmest frame of the *separable* windows (w018 supplies it, at
0.02024), giving **0.018212**.

| | result |
| --- | --- |
| empty-room frames skipped | **95.1%** (w022 95.9%, w023 94.3%) |
| occupied frames skipped, 21 separable windows | **0.0000%** — zero, in every one |
| w021 frames skipped | 86.1% |
| all frames skipped, pooled over 24 windows | 11.5% |

`src/helpers/calibrate_energy_gate.py` exits non-zero if the gate ever reaches a
separable window's occupant, and prints the unprotectable ones rather than
hiding them. `src/data_loader.calculate_energy_gate_from_files` implements the
same rule per fold, from training windows only, and writes
`src/models/best_energy_gate.npy` — the gate is a calibration artifact exactly
like p99, never a literal in `code.py`.

**Why gating w021 is defensible, and where the honest doubt is.** w021's seated
cell ranks 203rd of 216 (Step 10) and the window scores F1 ~0.006: the model
finds essentially nothing there whether or not it is invoked, so skipping 86% of
its frames forfeits almost no recall that exists. The empty-room half is a
genuine gain — the same model fires above 0.40 on 1008 of 7500 frames of w022,
and a skipped frame cannot produce a phantom. **But the doubt is real and should
be stated: if the unseen test session contains a seated-only recording in which
the subject IS above the empty-room floor, this gate will suppress a subject the
model could otherwise have found.** The 0.9 safety margin is the only hedge, and
it is a margin measured on 24 windows of one session.

**Interaction with 4.3, which is the reason the gate is not free.** Persistence
integrates the heatmap over ~10 s. Feeding it zeros for a gated frame decays that
integrator, which is correct for an empty room and is exactly wrong for a seated
subject the gate should not have touched. The two features therefore share one
failure mode — w021 — from opposite ends: 4.3 recovers seated evidence the decode
was throwing away, and 5.1 declines to generate evidence at all. They are scored
together, on the same OOF cache, for that reason.

### Step 12 — K1: the honest out-of-fold float-vs-int8 gap

*2026-08-29, on Christian's machine.* K1 had only ever been measured in-sample,
because the six fold models were discarded after CV and the deployed model is
the only one ever quantized. This branch's retrain caches **both** `preds`
(int8 path, what deploys) and `preds_float` (Keras) per fold, from
`logs/20260829-181356`, so the gap is finally measurable out-of-fold.

Scored with the actual deployed decode chain (`Persistence` → `extract_peaks_from_grid`
→ `Tracker`) and the official Hungarian matcher from `evaluation/evaluate_performance.py`
— not a reimplementation of either — swept across threshold:

| threshold | int8 F1 | float F1 | gap (float − int8) |
| --- | --- | --- | --- |
| 0.30 | 0.8201 | 0.8752 | +0.0551 |
| 0.35 | 0.8443 | 0.8809 | +0.0366 |
| 0.40 | 0.8605 | 0.8827 | +0.0223 |
| 0.45 | 0.8673 | 0.8792 | +0.0120 |
| 0.50 | 0.8682 | 0.8709 | +0.0026 |
| 0.55 | 0.8636 | 0.8545 | −0.0091 |
| 0.60 | 0.8507 | 0.8269 | −0.0238 |
| 0.65 | 0.8281 | 0.7810 | −0.0471 |

**int8 argmax: F1 0.8682 @ th=0.50** (matches `best_threshold.npy` exactly —
the training run picked its own true optimum). **Float argmax: F1 0.8827 @
th=0.40.** **Gap at each path's own best threshold: +0.0145**, float ahead —
consistent in direction and rough magnitude with the earlier in-sample proxy
(0.0080–0.0101 at th=0.40, widening to 0.037 at th=0.20 on the refit model).
K1 is closed: the two curves peak at different thresholds exactly as the
in-sample measurement predicted, and the int8 path is what `code.py` decodes
at, which is why the deployed `THRESHOLD` is 0.50 and not 0.40.

One secondary finding worth recording: the gap **flips sign above th≈0.52** —
int8 slightly outperforms float at higher thresholds in this measurement. Not
investigated further; the load-bearing result is the argmax comparison above.

Machine-dependence caveat (K8): this is one retrain, one machine. The direction
and rough size of the gap is what should be expected to reproduce; the exact
numbers are not guaranteed to.

### Step 13 — Phase 2: the real sweep ran. A promising, boundary-caveated signal, not adopted

*2026-08-29, on Christian's machine, `logs/20260829-181356/oof` (this branch's
int8-scored, 4.3+5.1-active cache).* 1008 configs (4 thresholds × 3
`min_distance` × 4 `alpha` × 7 `max_coast` × 3 `v_max`), 24 windows, 9.4 min.
Full table: `report/postproc_sweep.md`/`.csv`.

**Top result:** threshold 0.55, `min_distance` 0.70, `alpha` 0.25–0.35,
`max_coast` 6, `v_max` 4.0 (→ `max_distance` 1.69) — pooled F1 **0.8820**,
empty-room phantoms **1.65%**. Against the deployed config's true F1 (0.8682 @
th=0.50, Step 12), that is +0.0138.

**Not adopted, for two stacked reasons:**

1. **The winner sits on the grid edge for five parameters at once.** Read
   individually against `SWEEP_AXES`'s own comments: `min_distance` (0.70) and
   `alpha` (0.25–0.35) are non-issues — both axes are already established flat
   over their whole range, confirmed again here (top-10 F1 spans 0.8813–0.8820
   across every tested `alpha`). `max_coast` (6) and `v_max` (4.0) are
   *deliberate* ceilings, chosen specifically to prevent the phantom-rate
   amplification a prior, wider search already produced once (the discarded
   `max_coast=25` recommendation) — the code comments say explicitly not to
   widen either without a written reason, and "the sweep wants it" is not one.
   **Only `threshold=0.55` is a genuine gap**: that boundary is this run's own
   chosen range (0.40–0.55, from `characterize`'s separation plateau), not a
   project ceiling, so the true threshold optimum may sit higher and this run
   does not resolve it.
2. **No paired SE was computed for this specific comparison.** The sweep
   reports point estimates and a leave-one-window-out winning count (17/24
   folds), not the paired jackknife SE this project's own adoption rule
   requires (discipline #6: adopt nothing below 1 paired SE, computed against
   the actual deployed config, not a point estimate). +0.0138 is in the
   plausible range of "real" given the pooled paired SE has held near ±0.0089
   across many prior comparisons on this dataset, but "plausible" is not
   "measured," and this project's whole discipline exists to not skip that
   step when the deadline is close.

**What this closes and what it leaves open.** Phase 2's exit criterion — a
documented sweep table — is met; its adoption criterion is not, and per this
project's standing rule that means the deployed post-processing constants
(`THRESHOLD=0.5`, `Tracker(alpha=0.4, max_distance=1.0, max_coast=2)`) stay as
they are. If anyone picks this back up: widen `--thresholds` past 0.55 (the
only genuinely open axis), then run `oof_breakdown.py --compare` for a proper
paired SE against the true deployed config before touching `code.py`.

---

## (Add new topics above the appendix as they accumulate — data/preprocessing,
model architecture, quantization, tracking, evaluation methodology, etc. Same
pattern: newest work first within a topic, corrections made in place and
dated rather than deleted.)

---

# Appendix — Measurement discipline

*Folded in from `TASKBOARD.md` when the task board was retired at delivery.
These are the rules the whole record above was produced under, and the numbered
items are cited by number from the comments in `src/` — so they are kept
verbatim rather than rewritten. Section 5 of the report is the short version.*

Rules we're holding to:

1. **Report the CV out-of-fold F1**, never the example-set F1. The example set
   is derived from training data and the refit model has seen it.
2. **Never cite remembered size/arena/op numbers** — re-run
   `evaluate_constraint.py`.
3. **Post-processing changes need no retraining.** Sweep them on the cached OOF
   heatmaps. Only training changes cost a CV run.
4. **Don't claim an improvement that wasn't measured.** Distinguish measured
   from reasoned in the report.
5. **After every retrain**, refresh `code.py`'s `P99_VALUE` and `THRESHOLD` from
   `src/models/`, then run the calibration guard before trusting any number.
6. **Adopt nothing on less than 1 SE — but use the PAIRED SE, not ±0.018.**
   The ±0.018 (re-measured ±0.0195) is the jackknife SE of ONE model's pooled
   F1. Two models scored on the same 24 windows are a *paired* comparison, and
   per-window difficulty cancels: the SE of the **difference** is roughly half
   as large (measured 2026-08-27 over the Phase 4 ablation — pooled ±0.0089,
   seated subset ±0.0183, walking ±0.0047). Using the unpaired number as the
   A/B bar is too strict and would hide real effects. Compute it with
   `logs/phase4-ablation/paired_jackknife.py`.
   This changes the bar, not the Phase 4 verdict: `mag+delta32`'s +0.0053 is
   still only 0.6 paired SE. It did change one reading — `mag+aoa` at −2.4 SE
   is a *confirmed* regression, not the "probably noise" the unpaired bar would
   have called it.
7. **Compare at a MATCHED operating point.** 4.1d's threshold argmax moved
   0.40 → 0.30. Scored at each model's own argmax, its aggregate seated recall
   looks like a gain (62.2% vs 60.3%); at a matched threshold the same pair is
   a 1.6–1.9 SE loss. The whole apparent gain was the lower threshold buying
   recall for everyone. Report both thresholds; never compare a tuned
   operating point against an untuned one.
8. **Per-identity seated recall is unstable across retrains, so quote its own
   paired SE.** The same `mag` model family gives w004 seated recall of 8.7%
   on one machine and 33.1% on the other; measured paired SE on aggregate
   seated recall is ±0.028–0.037, roughly 3–4× the pooled-F1 SE. Compare only
   paired, same-machine, same-tool runs
   (`oof_breakdown.py <new> --compare <baseline>`), and treat any single
   window's seated number as one draw.
9. **A per-cell d′ is a feasibility check, not a prediction.** The Phase 4
   ablation was motivated by a d′ measured on the single best-SNR cell of one
   window. It was real and it reproduced, and the feature still failed, because
   the model sees all 846 cells and the other 845 carry the noise. Use such a
   measurement to rule things OUT cheaply; never to promise a gain.
10. **Regenerate the baseline; never reuse a recorded number as the paired
    reference.** 4.1b retrained its `mag` baseline on the same machine with the
    same code immediately before comparing — ~30 min that reproduced 0.8768 and
    73.2% exactly and, as a side effect, proved the channel-set infra, the
    radar-subset infra and the K=1 oversampling path are bit-exact no-ops
    (24/24 OOF caches byte-identical to `logs/20260826-122746/oof`). When the
    bar is ~1 SE on an unstable metric, the baseline is the wrong place to save
    time.
11. **For any data-side change, assert the split at FRAME granularity, against
    the real fold construction.** A grouped-CV leak does not throw, does not
    warn and does not look wrong in any output — it just inflates the score for
    the wrong reason. `src/helpers/verify_oversample.py` is the pattern: check
    the actual index array the loss iterates over, not a re-implementation of
    it, or you only prove that two copies of the same bug agree.
12. **Ask whether a change beats the THRESHOLD KNOB before asking whether it
    beats the baseline.** 4.1b's oversampled model was strictly dominated —
    lower seated recall, more phantoms, lower pooled F1 — by the *baseline at a
    lower threshold*, which costs no retraining at all. That is a
    cross-threshold check and no substitute for the matched comparison in #7,
    but it answers a decisive separate question: did this buy anything free
    post-processing could not.
13. **A synthetic test validates the implementation, not the premise.** 4.1c's
    `breath` channel passed every pre-CV check — passband within 0.01 Hz, a
    0.3 Hz cell reading 87× an equal-amplitude 2 Hz cell — and the check was
    still worth it, because it caught a float32 NaN divergence that would have
    corrupted the channel 150 s into every window. But it could not predict the
    failure, which was in the *modelling assumption*: real walking is a
    broadband 1–3 s transient with large in-band energy, not the out-of-band
    sinusoid the synthetic stood in for. **Pair every synthetic check with one
    histogram of the finished feature over real data of each class** — seconds
    of work, and it is what would have caught 4.1c before two CV runs.
14. **Verify a filter's numerics at the dtype you will actually run it at.**
    4.1c's bandpass has poles at radius 0.9877: stable in float64 and as a
    float32 biquad cascade, but the float32 direct form diverges to **NaN at
    frame 3743** of a 7500-frame window. None of that is visible in the
    frequency response, and the resulting NaN would not have looked like a
    filter bug anywhere downstream. **Refined by 4.1e: the criterion is
    direct-form TAP CANCELLATION, not pole radius.** `perio`'s poles sit at
    0.9967 — *closer* to the unit circle — and it is perfectly stable in
    float32, because a one-pole EMA's taps cancel by 599× against the
    bandpass's 3.5e8×, and float32 resolves ~1e-7.
15. **Score a candidate channel AFTER p99 normalization, in int8 levels.** The
    model is fed `clip(x, 0, p99)/p99` with ONE p99 pooled over all training
    windows, so a statistic that walking windows blow out reaches the seated
    subject as ~0 however well it separates in absolute terms. Measured on the
    median w021 seated cell: `mag` **10.6** levels, `breath` **20.1**, `perio`
    **0.8**. `breath` bought a real +1.2–2.1 SE seated gain on 20.1; 0.8 is 13×
    *less* than the plain magnitude baseline and cannot buy anything. Every
    Phase 4 step before 4.1e measured d′ or raw separation on the
    *unnormalized* channel — this is the cheap check none of them ran, and it
    is what let 4.1e cancel a CV run instead of spending one.
16. **Never pool a per-class statistic across mixed-class windows without
    checking for cross-contamination.** 4.1e's pooled walking/seated ratio was a
    benign 1.53× and its clean one was 57.6×. A range bin is a *shell* around a
    radar, so a walker crossing that shell dumps its much larger energy into a
    seated subject's own cell — inflating the "seated" figure ~40× in mixed
    windows (w004 411, w010 538, against seated-only w021's 10.61). The pooled
    number would have passed the gate and sent a doomed channel to a CV run.
    Use the single-class windows as the reference whenever they exist.
17. **Make the pre-run gate a program that EXITS NON-ZERO, not a section of
    printed numbers.** 4.1c's verifier printed plenty; what it lacked was a
    check that could *fail*. Everything 4.1c needed to know was computable
    before its first run and nobody computed it. 4.1e's gate cost ~4 minutes and
    cancelled a ~45-minute run, which is the whole argument.
    **Corollary added by 4.1f, and it is the expensive half of this rule: a gate
    only protects the columns it THRESHOLDS.** 4.1f's gate printed the number
    that predicted its failure and asserted on a different one. A printed number
    nobody thresholds is documentation, not a gate — and a passing verdict then
    reads as evidence when it is only silence. **Also: check that every check
    actually runs.** 4.1e's check 4 sat behind an early `return` as dead code
    while the verdict block reported it "PASS (asserted)" — an assertion that has
    never executed is worse than no assertion, because it is *trusted*.
18. **Gate a candidate channel on its EMPTY-CELL PEDESTAL, not only on its
    occupied-vs-empty contrast.** 4.1f delivered 177.9 int8 levels of seated
    contrast against `mag`'s 7.4 — a 24× better per-cell feature — and produced
    the largest regression in Phase 4, seated recall included. Its empty cells
    normalize to **0.302 of full scale against `mag`'s 0.018**, because an
    amplitude-invariant statistic cannot express "nobody is here" and
    uncorrelated noise self-correlates at ~1/√N_eff. At any instant **845 of 846
    cells are empty** and the early-fusion `Dense` sees every one of them, so a
    high-pedestal second channel is a nuisance plane occupying half the input
    dimensionality. **A candidate needs BOTH: contrast at the target AND a
    background no worse than the channel it is joining.** This is discipline #9
    ("the other 845 cells carry the noise") in the case where the per-cell
    measurement was not marginal but spectacular.
19. **Prefer an intervention's OWN control to the deployed baseline whenever it
    moves more than one knob.** 4.1d moved channels and radar count together and
    was recorded as a −9.1 SE regression; against the matched 4-radar control it
    is −0.5 SE, i.e. its channels did essentially nothing and the radar reduction
    carried the whole effect. The control cost one CV run, sat un-run for two
    days after 4.1d's own spec identified it, and corrected a headline number.
    **Relatedly, "geometrically free" is not "free":** dropping to 4 radars
    creates zero seated blind spots and still costs **−0.0688 pooled F1
    (−10.7 SE)**, comparable to the entire seated ceiling. Redundant coverage is
    worth a lot even where it is not needed to see a subject at all.
