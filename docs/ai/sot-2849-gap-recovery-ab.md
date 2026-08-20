# SOT-2849 — Node-interpolation gap recovery on the linking stage (leak-free CV A/B)

**Cycle 8, follow-on** to SOT-2848 on the biohub-claude Kaggle rank-improvement ladder
(parent SOT-2846). Loads the frontier upper-solution **linking-stage** win onto our pipeline.
Champion `detect-link-dog-v4-shorttrack` **byte-frozen** (CV micro-adj 0.6649). No champion /
registry / embedded mutation. **No Kaggle submission** (per issue).

## Axis screen (issue step 1)

The issue offered (a) learned linking (`SimpleNodeTransformer`) **or** (b) gap-recovery
post-processing, to be A/B'd on the SOT-2848 learned detection.

- **(a) is infeasible here.** SOT-2848 established the learned `TemporalUNet3D` detector has
  **no family-robust operating point** on this sparse GT (leak-free CV micro-adj **0.0**). A
  learned linker needs detected nodes to associate; with ~0 valid detections there is nothing to
  link, and an A/B "learned-detection + learned-linking" vs "learned-detection" is a degenerate
  0.0-vs-0.0 comparison. So the linking axis is screened on the **champion classical detection**
  (our real best pipeline), where a linking A/B is measurable. *(Interpretation disclosed on the
  Linear issue — safe default per design §2/§66.)*
- **(b) skip-edge gap-closing was already rejected** (SOT-2763): the competition edge metric keeps
  **only consecutive-frame edges** (`t_target − t_source == 1`), so a non-consecutive bridge is
  dropped before scoring and can never recover an FN edge.

The **one untested, mechanistically-distinct** frontier linking lever (the `pilkwang`
public-≈0.890 differentiator) is **gap recovery via node interpolation**: when reconnecting a
fragment tail@`t` to a head@`t+g`, **insert interpolated detection nodes** at each missing frame
`t+1 … t+g−1` (linear interpolation of the voxel centroid) so the recovered path is a chain of
**consecutive** edges. Those edges ARE scored, and an interpolated node landing within the ≤7 µm
per-timepoint match radius of the true (missed-detection) GT node recovers a **real FN edge**.

## Implementation (default-off, non-destructive)

`link.py`: new `_gap_recover` + `LinkParams.gap_recover` (default `False` → champion byte-for-byte)
with `gap_recover_max_gap` / `gap_recover_distance` / `gap_recover_min_frag`. Runs **before**
short-track pruning (a recovered bridge can also lift two real short fragments into one surviving
`≥ min_track_length` component). Bridges chosen by an optimal per-component min-cost assignment
(same decomposition as `_gap_close`). `gap_recover_min_frag` gates eligibility on the
weakly-connected fragment size at **both** terminals, to refuse the short-noise-fragment
resurrection that sank SOT-2763. Wired through `champion_params` (absent JSON keys ⇒ off).

## Evidence — SOT-2761 leak-free CV, champion DoG-v3 frozen, single-variable same-seed A/B

Detection cached once per family; every link variant re-linked off the cache and scored through the
one CV aggregation (byte-comparable to the registry champion 0.6649). Grid
`gap_recover_max_gap ∈ {2,3}` × `gap_recover_distance ∈ {7,10} µm` × `gap_recover_min_frag ∈
{1,3,4,5,6,8}` (24 variants).

**The mechanism works — the first linking axis in the ledger to recover real FN edges.**
Best `mg=3 / dist=7 / min_frag=5`:

| family | champion adj | A/B adj | ΔTP | ΔFP | ΔFN | Δnodes | reading |
|---|---|---|---|---|---|---|---|
| 44b6_0113de3b | 0.8895 | **0.8889** | 0 | 0 | 0 | +157 | **pure node-count penalty — zero edge benefit** |
| 44b6_0b24845f | 0.6817 | 0.7174 | +2 | 0 | −2 | +487 | real FN recovery |
| 6bba_05b6850b | 0.5700 | 0.5827 | +12 | −6 | −12 | +105 | real FN recovery + FP drop |
| 6bba_05db0fb1 | 0.7310 | 0.7424 | +15 | −2 | −15 | +925 | real FN recovery + FP drop |
| **micro-adj** | **0.6649** | **0.6773** | **+29** | **−8** | **−29** | | **Δ +0.0124** |

## Decision — NON-PROMOTION (champion byte-frozen)

**No single family-invariant operating point clears the per-dataset no-regression gate.** The sole
blocker is the ultra-sparse **44b6_0113de3b** (~1 GT cell/frame; the champion already links it near
perfectly at 0.8895): it recovers **zero** edges at **every** setting (`ΔTP=0` / `ΔFN=0` across all
24 variants) and pays only a small node-count penalty — monotone-improving with `min_frag`
(0.8878 → 0.8892 at `min_frag=8`, Δ **−0.0003**) but asymptoting **below** the floor unless the
mechanism is turned fully off. This is the same "no global operating point" wall as SOT-2830 /
SOT-2848, but qualitatively milder: there is **no edge corruption anywhere** (unlike SOT-2763's
matching corruption) — the only cost is a node-count penalty on a family with no recoverable gaps.

Given the established **local-CV ↔ LB divergence** (SOT-2816: CV rose 0.6232 → 0.6649 while public
LB fell 0.624 → 0.509), a +0.0124 CV micro gain is **not** a safe basis to disturb the byte-frozen
champion. So: champion `detect-link-dog-v4-shorttrack` `champion/config.json` + `registry.json` +
`EMBEDDED_CHAMPION_CONFIG` **unchanged**; `gap_recover` shipped **default-off** (documented knob,
exactly as SOT-2763 `max_frame_gap` / SOT-2818 `division_overlay` were kept after rejection).

**Reserve / late-window LB-probe candidate (top priority).** This is a *stronger* held candidate
than SOT-2840's global-MCF (θ=6.5): +0.0124 vs +0.0022 CV micro, genuine FN recovery vs mere
non-regression, and zero edge corruption. Effective candidate config: champion link block +
`gap_recover:true, gap_recover_max_gap:3, gap_recover_distance:7.0, gap_recover_min_frag:5`. Queue it
for a reserve-budget LB probe in the deadline's final window (deadline 2026-09-29), since only a real
LB read can resolve the CV↔LB divergence for a micro-positive / gate-failing candidate.

## Gates

- 173 pytest pass (169 prior + 4 new `tests/test_gap_recover.py`).
- `eval.cv --check-champion` micro-adj delta **0.0000** (champion byte-frozen).
- `champion_params` round-trip confirmed: absent keys ⇒ `gap_recover` off; enabled config flows
  through to `LinkParams`.
- Artifacts: `experiments/sot2849/{screen,refine}_gap_recover.json` + `.log`; ledger appended.

## Implication for the ladder

Node interpolation is the first lever that **turns detected-track continuity into scored FN-edge
recovery** — it is not exhausted, only gated by one structurally-saturated sparse family. The
frontier gain likely needs this **plus** family-adaptive gating (or dense pseudo-labels that give
the sparse family recoverable gaps), rather than the bare learned detector/linker.
