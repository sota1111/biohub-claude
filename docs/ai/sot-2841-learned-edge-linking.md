# SOT-2841 — GT-learned edge-linking cost (label = edge)

**Result: REJECTED (with evidence). Champion byte-frozen at 0.6649. Kaggle not submitted.**

## Axis

The champion `detect-link-dog-v4-shorttrack` links each `t → t+1` transition on
**scaled centroid distance alone**, so in the dense `6bba` families it can attach the
wrong, merely-nearer successor. The SOT-2828 / SOT-2829 hand-offs named **learned
LINKING (`label = edge`)** as an *untried* axis with a different, relative-and-dense
label space than the two rejected siblings:

- **SOT-2828** learned a *node* scorer from the sparse GT → positive-unlabeled
  contamination (an unmatched candidate may be a real un-annotated cell), zero LOFO
  discrimination, REJECTED.
- **SOT-2829** added *one* hand-crafted appearance-similarity term → a single
  non-learned feature, non-discriminating, REJECTED.

This child fits a light logistic classifier **leak-free (leave-one-family-out)** on
GT *consecutive* edges (positives) vs. the feasible non-GT successors of **matched**
sources (confident negatives — a matched source has a definite GT successor, so its
other feasible successors are true negatives, no node PU). The joint **edge**-feature
vector (`biohub_tracking.edge_linker.EDGE_FEATURE_NAMES`) is:

`[dist_scaled, app_cos (SOT-2829 descriptor cosine), src_rivals, dst_rivals,
succ_rank, pred_rank, succ_margin]`.

The learned `p_edge` enters the link cost as `dist + weight · (1 − p_edge)` —
**re-rank only, metric-valid**: the `≤ max_distance` feasibility gate stays on the
raw scaled distance, so it reorders the champion's existing feasible set and never
admits a new long-range edge. Kernel-safe (numpy-only, embedded coefficients,
sklearn-free inference, `exec()`/no-`__file__`). `weight = 0` / `edge_cost_model =
None` / no descriptors ⇒ champion **byte-for-byte** (strict default-off superset).

## Screen (`experiments/sot2841/screen_edge_linker.py`)

Leave-one-family-out: each held-out family is linked by a model trained **only** on
the other three families' `(edge-features, edge-labels)`, and the four held-out
scores aggregate into the SOT-2817 re-anchored full-metric leak-free CV.
Single-variable same-seed A/B; detection + descriptors cached once per family.

| weight | micro-adj | Δ score | dTP | dFP | dFN | no-regress | promotable |
|-------:|----------:|--------:|----:|----:|----:|:----------:|:----------:|
| 0.0 | 0.6649 | +0.0000 | 0 | 0 | 0 | (baseline) | — |
| 0.5 | 0.6649 | −0.0000 | 0 | 0 | 0 | False | False |
| 1.0–6.0 | 0.6642 | −0.0007 | −1 | +1 | +1 | False | False |

**Held-out LOFO positive-vs-negative `p_edge` gap (discriminative power):**

| family | p_pos | p_neg | gap | n_pos / n_neg |
|--------|------:|------:|----:|---------------|
| 44b6_0113de3b | 0.691 | 0.216 | **0.475** | 48 / 27 |
| 44b6_0b24845f | 0.753 | 0.356 | **0.398** | 37 / 11 |
| 6bba_05b6850b | 0.773 | 0.421 | **0.352** | 695 / 320 |
| 6bba_05db0fb1 | 0.634 | 0.193 | **0.441** | 1021 / 575 |
| **mean** | | | **0.416** | |

## Why rejected (root cause)

The learned edge model is **genuinely discriminative** — the first
linking-disambiguation axis with real held-out separation (mean LOFO gap 0.42 across
all four families, vs. SOT-2828's zero and SOT-2829's non-discrimination). **But the
discrimination does not convert to a CV gain.** At every positive weight the only
family that moves is the densest `6bba_05db0fb1` (dTP −1 / dFP +1 / dFN +1), a
−0.0007 regression; the other three families stay byte-identical.

Near-distance and being-the-GT-edge are highly correlated, so the classifier's
top-ranked successor is almost always **also** the nearest one — the champion's
distance-only optimal assignment already sits on that agreement, leaving essentially
no feasible-set headroom for a learned re-rank. The one edge it flips in the densest
family is a net loss. **The linking-disambiguation lever is saturated at the
champion.**

This is a *rejected*, not *inconclusive*: byte-invariance confirmed, single-variable
same-seed A/B across eight weights, and the LOFO gap measured as evidence (no
false-REJECT). Reverted to default-off; the module, tests, and screen are kept as
default-off infrastructure for any future cross-hop coupling term.
