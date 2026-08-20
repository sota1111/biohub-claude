# SOT-2830 — Portable global short-window min-cost-flow / birth-death-arc linking

**Cycle:** biohub-claude Kaggle 順位向上 第6次 (parent SOT-2827), escalation-ladder step 6
(external-knowledge / top-solution port).
**Result:** implemented (default-off); **DO-NOT-PROMOTE this cycle** — champion
`detect-link-dog-v4-shorttrack` kept **byte-frozen** (CV 0.6649). Best CV candidate
recorded for LB validation. **No Kaggle submission.**

## Axis

Port the top-solution common winner — *tracking-by-assignment* (ILP / min-cost-flow
global optimisation; arxiv 2004.06375 primal-dual, arxiv 1705.03386
proposal-selection) — as a **portable, pyscipopt-free** analog. The champion links
each `t → t+1` transition with an *optimal but greedy* nearest-neighbour bipartite
matching that attaches **every** feasible pair within `max_distance` (implicit
birth/death cost = ∞). Top solutions instead add explicit **birth/death arcs** so the
solver can *refuse* a marginal link rather than attach a transient detection.

## Implementation (`src/biohub_tracking/link.py`)

New `LinkParams` fields, all default to the champion (per-frame) behaviour:

- `global_window: int = 1` — `1` runs the **unchanged** per-frame path (champion
  graph **byte-for-byte**). `≥ 2` activates the global birth/death-arc path
  (`_global_link` / `_global_assign`).
- `birth_cost: float = inf`, `death_cost: float = inf` — birth/death arc costs; only
  their sum `θ = birth_cost + death_cost` enters, as the **link-acceptance
  threshold**: a `t → t+1` pair links only when its scaled distance is `< θ` (and
  `≤ max_distance`). `θ = ∞` ⇒ every feasible link kept ⇒ **reproduces the champion
  matching exactly** (`_global_assign` delegates to the per-frame `_assign`).

Design properties (all covered by `tests/test_link.py`):

- **scipy-only, pyscipopt-free** (networkx is not offline-bundled). Each transition
  is the exact single-transition min-cost flow, solved as an
  assignment-with-birth/death-outliers on a rectangular `n_src × n_dst`
  `linear_sum_assignment` (same size/cost as the champion — no dummy-padded square
  blow-up). Exec-compatible (`submit/exec_compat_gate.py` green).
- **Metric-valid — only consecutive `t → t+1` edges** are emitted; **no bridge/gap
  edge**, so this does *not* reintroduce the gap-closing (SOT-2763) non-continuous
  metric failure. In-linker division and gap-closing are inactive on the global path.
- **Default byte-invariant** — champion config has no `global_window` key ⇒ default 1
  ⇒ per-frame path unchanged; `champion/config.json`, `registry.json`, the embedded
  champion, and the CV reference constants are **untouched**.
- **Window is a structural knob (output-invariant here).** For the champion's
  pure-distance edge cost the window min-cost flow *decouples per transition* (a
  middle detection's in/out links share no flow variable; birth/death are per-node
  constants). So the joint W-frame optimum equals the per-transition optimum, and the
  effective lever is `θ`. `window ∈ {2,3}` verified identical (empirical decoupling +
  no bridge leakage). The value is reserved for a future cross-hop coupling term
  (velocity/appearance).

## A/B — screen → confirm (SOT-2817 re-anchored full-metric leak-free CV)

Frozen champion detection (`mad_k=3.0`), computed once per family; same-seed
deterministic A/B. Baseline reproduced **0.6649** exactly; `θ=∞` sanity reproduced
the champion **edge-for-edge** (0 edge delta); `window`-invariance confirmed.

θ (=birth+death, µm) sweep at window=2 — `experiments/sot2830/screen_global_mcf.json`:

| θ | micro_adj / score | Δ vs champion | ΣΔTP | ΣΔFP | ΣΔFN | per-dataset non-reg | family_mix_sensitive |
|---|---|---|---|---|---|---|---|
| ∞ (champion) | 0.6649 | +0.0000 | 0 | 0 | 0 | — | True |
| 6.5 | 0.6671 | **+0.0022** | −11 | −17 | +11 | **True (4/4)** | **True** |
| 6.0 | 0.6694 | +0.0045 | −21 | −37 | +21 | False | True |
| 5.5 | 0.6689 | +0.0040 | −38 | −56 | +38 | False | False |
| 5.0 | 0.6610 | −0.0039 | −76 | −76 | +76 | False | — |
| 4.5 | 0.6547 | −0.0102 | −114 | −102 | +114 | False | — |
| 4.0 | 0.6439 | −0.0210 | −167 | −131 | +167 | False | — |
| 3.5 | 0.6095 | −0.0554 | −268 | −144 | +268 | False | — |
| 3.0 | 0.5661 | −0.0988 | −409 | −186 | +409 | False | — |

**Mechanism confirmed.** Lowering `θ` trims the longest feasible links first, removing
net-FP mislinks (ΣΔFP < 0). At `θ=6.5` this removes 17 FP for 11 lost TP → adjusted
Jaccard rises. This is exactly the axis hypothesis (birth/death arcs suppress marginal
mislinks to transient detections). Below ~5.0 too many real TP edges are cut and both
micro and every family regress.

**θ=6.5/window=2 is the sole CV-promotable point** (+0.0022, **4/4 datasets
non-regressing**, and micro 0.6649→0.6671 / lineage-macro 0.7216→0.7272 / macro
0.718→0.7233 **all up**). Confirm (`confirm_global_mcf.json`) reproduced it exactly and
re-verified champion byte-invariance (CV 0.6649).

## Decision — DO-NOT-PROMOTE this cycle (champion byte-frozen)

Despite passing the raw no-regression bar, θ=6.5 is **`family_mix_sensitive=True`**
(micro↔lineage-macro gap **0.0601 > 0.05** tol) and its gain is concentrated on the
dominant **6bba** lineage (weight-share 0.9577; the gain is 6bba_05db0fb1 ΔFP −15 /
ΔTP −9 plus 6bba_05b6850b ΔFP −1). Per the **SOT-2817 representativeness guard** and
the **SOT-2816 CV-up/LB-down hazard**, a family-mix-sensitive dominant-lineage micro
gain is insufficient to flip the champion pointer **without LB validation** — and this
child cycle **does not submit**, so the LB oracle (the primary KPI) cannot settle it
now. This is the same discipline that rejected SOT-2818's (smaller) family-mix-sensitive
dominant-6bba gain.

So: champion `detect-link-dog-v4-shorttrack` stays **byte-frozen** (`--check-champion`
reproduces 0.6649); the global-MCF linker ships **default-off**; θ=6.5/window=2 is
recorded as the **best CV candidate** and handed to the next submission/converge cycle,
which can validate it on the LB before any pointer flip. Unlike every prior linking axis
(SOT-2762/2763/2818, all REJECTED with regressions), this is the **first no-regression,
all-aggregation-consistent** linking candidate — a live lever, pending LB confirmation.

## Acceptance criteria

- [x] 短窓大域割当リンクが scipy のみで実装され exec互換(pyscipopt非依存)
- [x] default で既存 per-frame リンクが byte不変・bridge edge を生成しない
- [x] 4データセットCVで同一seed A/B(sweep)結果を提示(否=非昇格、evidence付きで台帳)
- [x] Kaggle提出をしていない
- [x] `docs/ai/experiment_ledger.jsonl` に結果を追記
