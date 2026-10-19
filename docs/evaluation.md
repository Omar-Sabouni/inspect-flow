# How runs are scored

The generator records the box and label of every part that is fully in view,
meaning at least 6 px inside the frame. Parts closer to the edge are drawn but
not labelled. A seed fixes the whole part schedule, so a nominal run and a
poor-light run with the same seed contain the same parts.

## Per frame

`evaluate_run` in `inspectflow/evaluation.py` matches detections to labelled
parts in each frame at IoU 0.3 or more. It uses a min-cost flow, so it first
maximises the number of matches and then their total IoU, and the result
doesn't depend on detection order. From the matches it reports detection
precision and recall, a four-class confusion matrix for the matched parts,
and how often a part's track ID changes between frames.

## At the gate

The gate is a vertical line a quarter of the way across the frame. A track
gets one decision, on the first frame its center moves from left of the line
to on or right of it. `evaluate_gate` matches each decision to the labelled
part with the highest IoU in that frame and sorts it into one of these:

- correct pass or correct reject
- wrong defect: rejected, but for the wrong reason
- escape: a defective part passed
- false reject: a good part rejected
- spurious: no part matches, or the part already had a decision

A part that crosses the line without any decision counts as missed.

"Right call" in the README is the share of parts at the gate with the right
pass or reject verdict. Wrong-defect rejects count as right calls; missed
parts count as wrong. Escapes and false rejects are reported separately
because they cost different things on a real line: one ships a bad part, the
other scraps a good one.

The CSV that the dashboard exports has the matched part and the outcome on
every row.

## Seeds

- 7–26: test scenes, the numbers in the README. The dashboard shows seed 7.
- 100–104: validation, used to pick the CNN's score threshold and border margin.
- 1000 and up: CNN training.

Accuracy numbers are exactly repeatable for a given seed. Timing depends on
the machine. `inspect-flow bench` regenerates
[results/benchmark.md](results/benchmark.md), and
`inspect-flow bench --threads 1` gives the single-core version.

## Flat-field correction

`experiments/flat_field_correction.py` tests the standard fix for uneven light
on the classical rules. It estimates the light field with a large median
filter on a quarter-size brightness image, then scales every pixel so the belt
returns to its normal brightness. Validation seeds 100–102:

| Lighting | Correction | Detection recall | Class accuracy | ms per frame |
| --- | --- | ---: | ---: | ---: |
| Normal | none | 99.9% | 100.0% | 2.7 |
| Normal | flat-field | 99.9% | 100.0% | 7.8 |
| Poor | none | 64.6% | 45.9% | 2.3 |
| Poor | flat-field | 88.9% | 42.7% | 8.0 |
| Poor | bilateral filter, then flat-field | 80.4% | 42.5% | 23.1 |

The correction finds more parts but classifies them no better. On the darkest
empty stretch of belt it raises the noise from σ 2.9 to σ 14.1 gray levels.
The belt passes the saturation threshold (about 51 against 50) and is held out
of the mask only by the brightness threshold (about 60 against 65), so the
noisy belt pixels next to each part join its mask. On seed 100 a good part's
box grows by a median of 12 px in width and 8 px in height, and 59% of good
parts end up outside the 18% size tolerance, against 16% without correction.
Denoising first with a bilateral filter costs 15 ms per frame and doesn't
help. Retuning the segmentation thresholds for corrected frames hasn't been
tried.
