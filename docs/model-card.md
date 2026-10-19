# CNN part detector: model card

The CNN detector in `inspectflow/models/part_detector.onnx` finds parts on the
conveyor and assigns each one of four labels: `normal`, `incorrect_dimensions`,
`wrong_color` or `missing_feature`. This card describes how it was built and
where it should not be trusted.

## Intended use

- Comparison with the classical rule pipeline on InspectFlow's synthetic
  conveyor scenes.
- A worked example of training a detector from a simulator, exporting it to
  ONNX and serving it on CPU without a deep-learning runtime.

It is not a production inspection model. It has never seen a real camera
image.

## Architecture

| Property | Value |
| --- | --- |
| Type | Single-stage, anchor-free, CenterNet-style |
| Parameters | 144,952 |
| Input | 640 × 320 BGR, scaled to 0–1. Other frame sizes are resized. |
| Output | 8 × 40 × 80 grid (stride 8) |
| File | ONNX opset 17, 572 KiB, float32 |
| Runtime | OpenCV DNN (`cv2.dnn.readNetFromONNX`), CPU |

The backbone has three stride-2 convolutions followed by dilated 3 × 3
convolutions (dilation 1, 2, 4, 8). The dilations give each output cell a
receptive field of about 280 px, enough to see a whole part and judge its size.
Two heads predict:

- one center heatmap per label (sigmoid), and
- a sub-cell center offset (sigmoid) and log width and height.

Decoding keeps 3 × 3 local maxima of the class-wise maximum heatmap above a
score threshold, then applies IoU 0.5 non-maximum suppression. Boxes that come
within 5 px of the frame edge are dropped, matching the ground-truth rule that
ignores parts closer than 6 px to the edge.

## Training data

All training frames come from `SyntheticConveyor` in the nominal scenario,
using scene seeds 1000 and above. Each frame then gets random photometric
damage, applied in this order:

| Step | Range |
| --- | --- |
| Left unchanged | 25% of frames |
| Global gain | 0.35–1.2 |
| Light field | Linear gradient (45%), radial vignette (30%) or none |
| White balance | ±8% per channel |
| Gaussian noise | σ = 0–20 |
| Gaussian blur | σ = 0–1.6 px |

The poor-lighting test scenario uses a moving Gaussian spotlight with flicker.
That light model is not in the training augmentation. Its noise and blur fall
inside the training ranges.

Loss: CenterNet focal loss on the heatmaps (α = 2, β = 4), L1 on offsets and on
log size (weight 0.5). Heatmap cells covering parts cut off by the frame edge
are masked out of the loss, because those parts have no label.

Optimizer: AdamW, learning rate 2e-3 with a one-cycle schedule, weight decay
1e-4, batch size 32, 4,000 steps. Training took 8 minutes on an RTX 4050 laptop
GPU. The full record is in `inspectflow/models/part_detector.json`.

## Data splits

| Split | Scene seeds | Used for |
| --- | --- | --- |
| Train | ≥ 1000 | Gradient updates |
| Validation | 100–104 | Choosing the score threshold and border margin |
| Test | 7–11 | Numbers in the README and `docs/results/` |

The web dashboard uses seed 7, so the dashboard shows a test scene.

## Results

Validation, 240 frames per seed, from the training record:

| Scenario | Detection precision | Detection recall | Verdict accuracy (matched parts) |
| --- | ---: | ---: | ---: |
| Nominal | 99.8% | 100.0% | 100.0% |
| Poor lighting | 99.7% | 99.98% | 100.0% |

On the 20 test scenes (seeds 7–26) the CNN made the right gate decision for
all 440 parts, 220 per lighting scenario. The full test table, with the
classical rules alongside, is in [results/benchmark.md](results/benchmark.md).

All 19 false positives on the validation seeds are parts whose true box starts
5 px from the left edge, 1 px short of the clearance that ground truth
requires. The CNN places them about 2 px further in, so the decoder keeps
them. None is a detection of empty belt.

## Limitations

- The parts are flat rounded rectangles on a plain belt. Real parts have
  texture, specular highlights, shadows and perspective, and the model has
  seen none of them.
- Each label was learned from a few fixed examples: four wrong sizes, three
  wrong colors and one missing hole. Defects outside those sets, such as a
  scratch or a fifth color, are untested. The classical rules check measured
  tolerances, so size and color deviations of any kind still fail them.
- Frames are resized to 640 × 320, so pixel sizes only carry their synthetic
  meaning at that resolution.
- Detection scores are not calibrated probabilities. On validation seed 100
  the median score is 0.99 and 95% of scores are above 0.94.

## Reproduce

```bash
python -m pip install -e .[train]
python training/train_detector.py --steps 4000 --device cuda
```

PyTorch's ONNX export is pinned to the TorchScript exporter at opset 17,
because OpenCV 4.x does not fully support the opset 18+ graphs that the newer
exporter emits. Retraining changes the weights and the SHA-256, so
`tests/test_detectors.py` checks that the model file and its record still
match.
