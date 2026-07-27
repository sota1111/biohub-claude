# Division and four-dataset detection sanity

SOT-2047 extends local validation in two directions: a deterministic synthetic
division makes the Division Jaccard term measurable, and the champion detector
is checked across every test dataset family.

## Synthetic division round trip

`inject_synthetic_division()` copies a linear graph, selects the earliest edge
with a complete parent/daughter/granddaughter window, and adds a second daughter
lineage 4 µm away along x. The offset is below the evaluator's 7 µm matching
radius and is converted to voxels with the supplied physical scale.

The golden tests cover all division-count outcomes:

| prediction | TP | FP | FN | Division Jaccard |
| --- | ---: | ---: | ---: | ---: |
| synthetic graph round-tripped against itself | 1 | 0 | 0 | 1.0 |
| original linear graph against synthetic GT | 0 | 0 | 1 | 0.0 |
| round-trip plus an extra evaluable fork | 1 | 1 | 0 | 0.5 |

Run the golden validation with:

```bash
pytest -q tests/test_synthetic_division.py
```

## Champion detection sanity

Run detection and linking directly from the four test images:

```bash
python scripts/detection_sanity.py \
  --data-dir data/test \
  --output docs/detection-sanity-all-test.json
```

When an already-generated champion submission needs to be audited without
redistributable image data, use:

```bash
python scripts/detection_sanity.py \
  --submission submission.csv \
  --output docs/detection-sanity-all-test.json
```

The committed record was produced from the deterministic `detect-link-v1`
champion submission. A dataset fails sanity if it contains a missing internal
frame or a frame whose detections exceed five times that dataset's median.

| dataset | nodes | track length min / median / max | detections per frame min / median / max | result |
| --- | ---: | ---: | ---: | --- |
| `44b6_0113de3b` | 12,269 | 1 / 5 / 62 | 104 / 122 / 137 | pass |
| `44b6_0b24845f` | 3,599 | 1 / 2 / 27 | 18 / 36 / 57 | pass |
| `6bba_05b6850b` | 3,405 | 1 / 6 / 65 | 30 / 34 / 40 | pass |
| `6bba_05db0fb1` | 5,433 | 1 / 3 / 39 | 88 / 118 / 138 | pass |

Both `6bba` datasets have non-zero detections in every frame and no explosive
frame. Full per-frame counts are recorded in
[`detection-sanity-all-test.json`](detection-sanity-all-test.json).
