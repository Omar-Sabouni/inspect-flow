# How InspectFlow scores a run

Every frame comes from `SyntheticConveyor`, which records the true box and
label of every fully visible part. InspectFlow compares its output with that
record at two levels: per frame and at the inspection gate. The gate-level
numbers are the ones a production line would care about.

## Ground truth

A part is labelled when its whole box is at least 6 px inside the frame. Parts
closer to the edge are drawn but unlabelled, and are listed as
`partial_boxes` so CNN training can ignore them.

Each seed fixes the part schedule:

- Labels are dealt in blocks of four, one per class in random order.
- Good parts vary by ±3 px in width, ±2 px in height and ±6 per color channel.
- Dimension defects use one of four shapes, each at least 25% off nominal on
  one axis. Color defects use one of three colors.
- Spawn times jitter by ±1 frame.

The same seed gives the same schedule in both scenarios. Only the lighting
differs, so nominal and poor-lighting runs compare like with like.

## Per-frame metrics

`evaluate_run` in `inspectflow/evaluation.py` matches detections to true parts
in each frame with IoU ≥ 0.3. The matching is a min-cost flow, which first
maximizes the number of matches and then the summed IoU, so it does not depend
on detection order.

| Metric | Definition |
| --- | --- |
| Detection precision, recall | Over all frames, from matched and unmatched boxes |
| Defect class accuracy | Correct labels ÷ matched parts (four-class confusion matrix) |
| Binary precision, recall, F1 | Defective vs normal, over matched parts |
| Track continuity | 1 − ID switches ÷ frame-to-frame transitions of matched parts |

## Gate metrics

The gate is a vertical line at 25% of the frame width. A track gets one
decision, on the first frame its centroid moves from left of the line to on or
right of it. `evaluate_gate` matches each decision to the true part with the
highest IoU in the decision frame and assigns an outcome:

| Outcome | Meaning |
| --- | --- |
| Correct pass | Good part passed |
| Correct reject | Defective part rejected for the right defect |
| Wrong defect | Defective part rejected for a different defect |
| Escape | Defective part passed |
| False reject | Good part rejected |
| Spurious | Decision with no matching part, or a second decision for one part |
| Missed | Part crossed the gate but got no decision |

The summary rates are:

| Rate | Numerator | Denominator |
| --- | --- | --- |
| Gate verdict accuracy | Right pass/fail verdicts: correct pass, correct reject, wrong defect | Parts at the gate (missed parts count as wrong) |
| Escape rate | Escapes | Defective parts at the gate |
| False reject rate | False rejects | Good parts at the gate |
| Miss rate | Missed parts | Parts at the gate |

Escapes and false rejects carry different costs. An escape ships a bad part. A
false reject scraps or reworks a good one. A real line would set the two
thresholds separately, so InspectFlow reports both counts rather than one
combined score.

The dashboard shows these metrics for every run. The CSV decision log adds
`truth_part_id`, `truth_defect` and `outcome` to every row.

## Seeds and repeatability

| Seeds | Role |
| --- | --- |
| 7–11 | Test. Reported numbers. The dashboard uses seed 7. |
| 100–104 | Validation. CNN threshold and border margin were tuned here. |
| ≥ 1000 | CNN training scenes |

Generation is byte-for-byte deterministic for a given seed, so accuracy numbers
repeat exactly. Latency numbers depend on the machine and its load.

Run the benchmark:

```bash
inspect-flow bench                 # all CPU threads
inspect-flow bench --threads 1     # closer to a small edge CPU
inspect-flow bench --json out.json # also keep per-seed rows
```

## Negative result: flat-field correction

Before comparing the rules with a CNN, it is fair to try the standard classical
fix for uneven light. `experiments/flat_field_correction.py` estimates the
light field with a large median filter and rescales each pixel so the belt
returns to its nominal brightness. On validation seeds 100–102 it gave:

| Lighting | Correction | Detection recall | Defect class accuracy | ms/frame |
| --- | --- | ---: | ---: | ---: |
| Nominal | None | 99.9% | 100.0% | 2.73 |
| Nominal | Flat-field | 99.9% | 100.0% | 7.81 |
| Nominal | Bilateral filter, then flat-field | 99.9% | 100.0% | 22.60 |
| Poor | None | 64.6% | 45.9% | 2.27 |
| Poor | Flat-field | 88.9% | 42.7% | 8.04 |
| Poor | Bilateral filter, then flat-field | 80.4% | 42.5% | 23.13 |

Correction raises detection recall by 24 points, but classification does not
improve. The gain that restores the belt's brightness also multiplies the
sensor noise: on the darkest empty-belt patch, noise rises from σ 2.9 to σ 14.1
gray levels. The belt already passes the saturation threshold (about 51
against 50) and is kept out of the part mask only by the brightness threshold
(about 60 against 65). Amplified noise lifts belt pixels around each part over
that limit, and they join the part's mask. On seed
100, the measured box of a good part grows by a median of 12 px in width and
8 px in height, and 59% of good parts fall outside the ±18% size tolerance
(16% without correction). Good parts that were rejected for color are now
rejected for size. A bilateral filter before correction costs 15 ms per frame
and does not help.

The correction is not in the shipped pipeline. Retuning the segmentation
thresholds for corrected frames might change the result. That has not been
tried.
