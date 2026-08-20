# SOT-2763 — gap-closing (欠測フレーム跨ぎ) linking: REJECTED

**Cycle 2 · axis: classical Jaqaman/TrackMate second-LAP gap-closing.**
Champion **MAINTAINED** at `detect-link-dog-v4-shorttrack` (CV micro-adj 0.6649).
No champion/registry/embedded mutation. No Kaggle submission (parent SOT-2757 only).

## Hypothesis

The champion links only consecutive frames (frame-to-frame Hungarian, 7 µm), so a
cell missed in a single detection frame ends its track — the standard source of FN
edges. The public "Learned Graph w/ Gap Recovery" solution and classical LAP
trackers (Jaqaman 2nd step / TrackMate gap-closing) recover these by bridging track
fragments across missing detections. Implemented as a CPU numpy/scipy second LAP
step it should recover FN edges and raise the score.

## Implementation

`link.py`: after frame-to-frame linking, `_gap_close` bridges each fragment **tail**
(node with no successor) at `t` to a later fragment **head** (node with no
predecessor) at `t + g` for a frame gap `2 <= g <= max_frame_gap`, within
`gap_distance` µm, by an optimal min-cost assignment (solved per connected component
of the feasible-pair bipartite graph → block-diagonal LAP == global LAP, tractable on
70k-node volumes). Runs **before** short-track pruning so a bridge can rescue real
short fragments into a `>= min_track_length` component. New `LinkParams` knobs
`max_frame_gap` (default 1 = off, byte-for-byte champion) and `gap_distance`
(default 7 µm), wired through `champion_params`.

### Metric constraint (decisive)

The competition edge metric keeps **only consecutive-frame edges**
(`edge_metric._sanitise_pred_edges` step 1: `t_target - t_source == 1`). A bridge
edge spanning a gap is dropped before scoring — never TP, never FP. So gap-closing
has **no direct edge lever**; its only possible effect is *changing which nodes
survive the min_track_length prune*, which then feeds the per-timepoint node matching.

## Evidence (SOT-2761 leak-free CV, DoG-v3 frozen, single-variable same-seed A/B)

Grid `max_frame_gap ∈ {2,3}` × `gap_distance ∈ {7,10,14} µm`, detection computed once
per family, re-linked per variant, scored through `eval.cv`. **Every variant
regresses; none passes the per-dataset no-regression gate.** Monotone with looseness
(tighter = less bad):

| max_frame_gap | gap_distance | micro-adj | Δscore | ΔTP | ΔFP | ΔFN |
|---|---|---|---|---|---|---|
| — champion — | — | **0.6649** | 0.0000 | 0 | 0 | 0 |
| 2 | 7 | 0.6509 | −0.0140 | −12 | +23 | +12 |
| 3 | 7 | 0.6461 | −0.0188 | −16 | +29 | +16 |
| 2 | 10 | 0.6462 | −0.0187 | −17 | +30 | +17 |
| 3 | 10 | 0.6359 | −0.0290 | −30 | +43 | +30 |
| 2 | 14 | 0.6379 | −0.0270 | −28 | +41 | +28 |
| 3 | 14 | 0.6323 | −0.0326 | −34 | +45 | +34 |

Per-dataset matched-edge accounting at the least-bad `mfg=2/gd=7` (adj vs champion
floor; ΔTP/ΔFP/ΔFN/Δpred_nodes vs champion):

| family | adj | floor | ΔTP | ΔFP | ΔFN | Δnodes | reading |
|---|---|---|---|---|---|---|---|
| 44b6_0113de3b | 0.8875 | 0.8895 | 0 | 0 | 0 | +560 | **pure node-count penalty** — bridges only stitched noise fragments carrying no evaluable edge |
| 44b6_0b24845f | 0.6764 | 0.6817 | 0 | 0 | 0 | +2538 | same: zero edge change, +2538 resurrected noise nodes |
| 6bba_05b6850b | 0.5433 | 0.5700 | −13 | +18 | +13 | +643 | **matching corruption** — resurrected nodes steal per-timepoint matches, real edges → FN, spurious → FP |
| 6bba_05db0fb1 | 0.7275 | 0.7310 | +1 | +5 | −1 | +1522 | +1 TP swamped by node-count penalty on +1522 nodes |

**The FN-recovery hypothesis is refuted.** Net ΔFN is **+12 (positive)** — gap-closing
recovered *no* real FN edges anywhere. On the two clean 44b6 families it recovered
zero edges (ΔTP=0) while resurrecting hundreds/thousands of nodes → pure node-count
penalty. On the dense 6bba_05b6850b it *actively* lost 13 TP by corrupting the ≤7µm
per-timepoint matching. The short-track prune was already correctly removing these
short fragments as noise (the champion's own SOT-2369 thesis: "a detection that never
links into a ≥4-node track is almost always noise"); gap-closing works *against* it by
resurrecting that noise. And because the bridge edge itself is non-consecutive, the
only alternative order (gap-close **after** prune, stitching already-surviving long
tracks) is a strict no-op on this metric (no node change, bridge edge dropped) — so
there is no admissible ordering that nets positive.

### Confirm (independent, real champion_params → run_pipeline)

Re-scored baseline + `mfg=2/gd=7` challenger through the real
`champion_params(config) → run_pipeline` + `evaluate_cv` (re-runs detection, not
cached): baseline 0.6649, challenger **0.6509 (Δ −0.0140, no_regression=False)** —
reproduces the screen. Guards that the new `max_frame_gap` / `gap_distance` fields
flow through `champion_params` and that the champion default stays `max_frame_gap=1`
(off). See `experiments/sot2763/confirm_gap_closing.json`.

## Decision

**REJECTED.** Champion maintained at `detect-link-dog-v4-shorttrack`;
`champion/config.json` + `registry.json` + `EMBEDDED_CHAMPION_CONFIG` unchanged. The
gap-closing mechanism (`link.max_frame_gap`, default 1 = off; `link.gap_distance`) is
kept as a documented default-off knob, exactly as the SOT-2369 motion lever
(`velocity_gain`) and the SOT-2762 over-split gate (`division_max_sibling_ratio`) were
kept after their rejections. 71 pytest + exec-compat gate green. No Kaggle submission.

Artifacts: `experiments/sot2763/{screen_gap_closing,confirm_gap_closing}.json`.
