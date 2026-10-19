# CNN part detector

`inspectflow/models/part_detector.onnx` finds parts on the belt and labels each
one `normal`, `incorrect_dimensions`, `wrong_color` or `missing_feature`. The
training record, with hyperparameters, losses and the file's SHA-256, is in
`part_detector.json` next to it.

| | |
| --- | --- |
| Type | CenterNet-style, single stage, no anchors |
| Parameters | 144,952 |
| Input | 640 × 320 BGR, scaled to 0–1 (other sizes are resized) |
| Output | 8 × 40 × 80: four class heatmaps, x/y offset, log width and height |
| File | ONNX opset 17, 572 KiB |
| Runtime | `cv2.dnn` on CPU |

## Network

Three stride-2 convolutions (16, 24 and 48 channels) bring the frame down to a
40 × 80 grid. Four more 3 × 3 convolutions with dilation 1, 2, 4 and 8 widen
the receptive field to about 280 px, so each cell sees a whole part and can
judge its size. Two small heads predict the class heatmaps and the box.

Decoding takes 3 × 3 local maxima of the highest class heatmap above 0.4,
drops boxes within 5 px of the frame edge, and runs NMS at IoU 0.5. The
threshold and margin were picked on the validation seeds.

## Training

Frames are generated on the fly from nominal scenes with seeds 1000 and up,
then damaged at random:

| Augmentation | Range |
| --- | --- |
| None | 25% of frames |
| Global gain | 0.35–1.2 |
| Light field | linear gradient (45%), radial vignette (30%) |
| White balance | ±8% per channel |
| Gaussian noise | σ 0–20 |
| Gaussian blur | σ 0–1.6 px |

The poor-light test scene uses a moving spotlight with flicker. No augmentation
models a spotlight.

The heatmaps use CenterNet's focal loss; offsets and log sizes use L1, the size
term weighted 0.5. Parts cut off by the frame edge have no label, so the loss
ignores the cells around them. AdamW with a one-cycle schedule peaking at
2e-3, batch 32, 4,000 steps: 8 minutes on a laptop RTX 4050.

## Results

| Scenes | Lighting | Precision | Recall | Class accuracy |
| --- | --- | ---: | ---: | ---: |
| Validation, seeds 100–104 | Normal | 99.8% | 100% | 100% |
| Validation, seeds 100–104 | Poor | 99.7% | 99.98% | 100% |

Class accuracy is over matched parts. All 19 validation false positives are
parts whose true box starts 5 px from the left edge, 1 px short of the 6 px the
ground truth requires.

On the 20 test scenes (seeds 7–26) the CNN made the right gate decision for
all 440 parts. The full table is in [results/benchmark.md](results/benchmark.md).

## Limits

- It has only seen flat, rounded synthetic parts on a plain belt.
- It learned four wrong sizes, three wrong colors and a missing hole. A defect
  outside that set, such as a scratch, has never been tested.
- Scores are not calibrated probabilities. On seed 100 the median is 0.99.

## Rebuild

```bash
pip install -e '.[train]'
python training/train_detector.py --device cuda
```

The export uses PyTorch's TorchScript exporter at opset 17, which OpenCV's DNN
module reads. `tests/test_cnn.py` checks that the model file still matches the
hash in its training record, and that the CNN still decides every part
correctly in a poor-light test scene.
