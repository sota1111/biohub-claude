# SOT-2048 — gap-closing linking screen

Consecutive-frame linking can permanently split a track after a missed
detection. This experiment tested deterministic, distance-gated optimal
assignment from tracklet endpoints to later tracklet starts across one or two
missing frames.

## Candidate

The candidate added `LinkParams.max_frame_gap` with a backwards-compatible
default of `0`. After normal consecutive linking, it processed tracklet starts
chronologically and optimally assigned eligible endpoints within the champion's
7 µm distance gate. Focused synthetic tests covered disabled behavior, one-frame
bridging, gap and distance limits, one-to-one optimal assignment, and invalid
parameters.

## Screen → confirm

Dataset: local GT `44b6_0113de3b`, using the incumbent detector and linker
configuration. Full machine-readable results are in
[`sot-2048-gap-closing-evaluation.json`](sot-2048-gap-closing-evaluation.json).

| max missing frames | predicted edges | edge TP/FP/FN | adjusted Edge Jaccard |
| ---: | ---: | ---: | ---: |
| 0 | 11,193 | 47 / 2 / 3 | 0.9512 |
| 1 | 11,374 | 47 / 2 / 3 | 0.9512 |
| 2 | 11,426 | 47 / 2 / 3 | 0.9512 |

The gap candidates added 181 and 233 edges respectively, but none recovered a
GT edge and neither changed evaluable FP. Confirmation of the selected baseline
reproduced exactly.

## Decision — not promoted

The predeclared gate required adjusted Edge Jaccard to improve without an edge
FP increase. Neither gap candidate improved the metric, so the candidate code
was reverted as required by the Issue. The champion remains unchanged with
consecutive-only linking (effective gap `0`); no config, registry, or kernel
change is warranted.

## Kaggle submission

Following the review instruction that every evaluated candidate must end with a
real submission even when it is not promoted, the unchanged champion was rebuilt
and submitted:

- Kernel: `sota1111/biohub-claude-champion`, version 4
- Kernel execution: complete
- Competition submission: `55033776`
- Submitted at: 2026-07-27 16:30:31 UTC
- Description: `SOT-2048 non-promotion validation: current champion,
  gap-closing disabled`
- Initial competition status: `PENDING` (submission accepted for scoring)

This submission deliberately represents the unchanged champion, not the reverted
gap-closing candidate. It proves the non-promotion path still produces and
submits an exec-compatible competition artifact while keeping effective
`max_frame_gap=0`.
