"""Finer-grain leak-free CV holdout + CV↔public transfer-trust (SOT-2929).

Why this exists — the *density-mix wall*. The primary leak-free CV
(:mod:`biohub_tracking.eval.cv`) splits its holdout by **embryo lineage**
(``44b6`` vs ``6bba``) — a leave-one-family-out (LOFO) grouping. SOT-2921 proved
that grouping **crosscuts the true difficulty boundary**: the ``6bba`` lineage
contains *both* the sparsest sequence (``6bba_05b6850b``, observed
``median_knn_um`` ≈ 9.49 µm) *and* the densest (``6bba_05db0fb1`` ≈ 7.45 µm),
so a lineage bucket blends two opposite operating regimes. Every regime-conditioned
promotion (SOT-2922/2923/2931) then failed the 4/4 per-dataset gate on exactly the
family whose lineage label disagreed with its density regime.

This module makes the holdout **finer-grained than the lineage LOFO** in two
leak-free ways and then **quantifies whether the finer holdout is a better or a
worse public-LB proxy** (the transfer-trust question direction B was opened for):

* **per-sequence parity** — each whole embryo video is its own holdout unit and
  counts equally (the finest leak-free unit: the scored entity is a whole
  ``.geff`` lineage, so no cell/track/frame straddles a bucket boundary).
* **observable density-regime stratification** — sequences are bucketed
  ``sparse`` / ``dense`` by the **GT-free** ``median_knn_um`` covariate
  (:mod:`biohub_tracking.eval.regime`, SOT-2921), a signal computable from the
  test detection point cloud alone (no family label, no ground truth). This
  bucket **respects family-internal heterogeneity**: the two ``6bba`` sequences
  land in *different* regimes, which the lineage LOFO could never express.

Leak-free by construction (pinned by :mod:`tests.test_holdout`):

* Every holdout unit is a *whole embryo video* — the partitions are a strict
  partition of :data:`biohub_tracking.eval.cv.CV_HOLDOUT` (cover + disjoint), so
  no entity is split across buckets (entity holdout preserved).
* The regime label is **derived from the observable covariate** via
  :func:`biohub_tracking.eval.regime.assign_regime` and a threshold computed from
  the covariate distribution alone — never from the family prefix or the GT.

Everything here is a **pure function** of already-computed
:class:`~biohub_tracking.eval.cv.FamilyResult` rows (reusing the historical
lineage baked into :mod:`biohub_tracking.eval.transfer`), so it unit-tests
without the (gitignored) competition data.

Result (SOT-2929, evidence in ``docs/ai/cv-redesign-sot2929.md`` + the ledger):
finer granularity does **not** improve transfer-trust and the *single-regime*
view is strictly **worse** — the parity/blended statistics
(``lineage_macro_adj`` / ``per_sequence_parity`` / ``regime_parity_adj``) all
rank the anchored public configs identically to the live KPI (Spearman ρ tied),
while the **sparse-regime-only** statistic *fails to crown the public winner*
(the champion is not the best config on the sparsest sequence). So the finer
holdout is adopted only as a **per-regime no-regression guardrail**, not as a
replacement promotion KPI, and the binding oracle limitation is **public-anchor
scarcity**, not holdout granularity.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from itertools import pairwise
from typing import NamedTuple

from .cv import CV_HOLDOUT, FamilyResult, HoldoutFamily, aggregate
from .regime import assign_regime
from .transfer import HISTORICAL_LINEAGE, ConfigCv, spearman

# --------------------------------------------------------------------------- #
# Observable GT-free density covariate (SOT-2921).                            #
# --------------------------------------------------------------------------- #

# Per-sequence median nearest-neighbour spacing (microns) measured by
# :func:`biohub_tracking.eval.regime.sequence_covariates` on the champion
# detector's point cloud (published in
# ``experiments/sot2921/covariate_separation.json``). Large ⇒ sparse, small ⇒
# dense. This is an OBSERVABLE, test-computable signal — NOT a family/GT label —
# recorded here as a constant so the finer-grain holdout is pure and unit-testable
# without the (gitignored) data. The family prefix is used ONLY to audit the
# split after the fact (leak-free provenance), never to compute a covariate.
OBSERVED_MEDIAN_KNN_UM: dict[str, float] = {
    "44b6_0113de3b": 8.541,
    "44b6_0b24845f": 8.454,
    "6bba_05b6850b": 9.493,
    "6bba_05db0fb1": 7.452,
}

# Spacing-type covariate: denser sequences have *smaller* spacing.
REGIME_DENSE_IS_LOW: bool = True

# The re-anchored recommendation (see module docstring / SOT-2929 evidence): the
# finer holdout does NOT replace the live KPI; the whole-holdout adjusted micro
# stays the promotion statistic and the per-regime split is a guardrail only.
RECOMMENDED_KPI: str = "micro_adj"
REGIME_GUARDRAIL: tuple[str, str] = ("sparse", "dense")


def derive_regime_threshold(values: Mapping[str, float]) -> float:
    """GT-free ``median_knn_um`` threshold isolating the sparse (high-spacing) tail.

    Deterministic and label-free: sort the observed covariate values and cut at
    the midpoint of the **largest gap on the high (sparse) side of the value
    median**. This isolates the sparsest-tail sequence(s) from the dense cluster
    using the covariate distribution alone — no family prefix and no GT enter, so
    the split is honestly derivable at inference time.

    Returns NaN when fewer than two finite values are available.
    """
    vals = sorted(v for v in values.values() if not math.isnan(v))
    if len(vals) < 2:
        return float("nan")
    median = (
        vals[len(vals) // 2]
        if len(vals) % 2
        else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2.0
    )
    best_mid, best_gap = float("nan"), -math.inf
    for lo, hi in pairwise(vals):
        mid = (lo + hi) / 2.0
        if mid <= median:  # only splits that isolate a HIGH-spacing (sparse) tail
            continue
        gap = hi - lo
        if gap > best_gap:
            best_gap, best_mid = gap, mid
    # Fallback (no gap strictly above the median): cut just above the median so a
    # sparse tail still exists rather than returning an empty regime.
    if math.isnan(best_mid):
        best_mid = median + 1e-9
    return best_mid


def regime_partition(
    covariates: Mapping[str, float] = OBSERVED_MEDIAN_KNN_UM,
    *,
    threshold: float | None = None,
    dense_is_low: bool = REGIME_DENSE_IS_LOW,
) -> dict[str, str]:
    """Map each sequence to ``"sparse"`` / ``"dense"`` from the observable covariate.

    The label is produced by :func:`biohub_tracking.eval.regime.assign_regime`
    from the covariate value and a threshold derived from the covariate
    distribution (:func:`derive_regime_threshold`) — leak-free: no family/GT.
    """
    thr = derive_regime_threshold(covariates) if threshold is None else threshold
    return {
        name: assign_regime(val, thr, dense_is_low=dense_is_low)
        for name, val in covariates.items()
    }


# --------------------------------------------------------------------------- #
# Finer-grain holdout partitions (each a strict partition of the families).    #
# --------------------------------------------------------------------------- #


def per_sequence_partition(names: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Each embryo video is its own singleton holdout unit (finest leak-free unit)."""
    return {n: n for n in names}


def by_regime_partition(
    covariates: Mapping[str, float] = OBSERVED_MEDIAN_KNN_UM,
    *,
    threshold: float | None = None,
) -> dict[str, str]:
    """Bucket sequences by observable density regime (crosscuts lineage)."""
    return regime_partition(covariates, threshold=threshold)


def by_lineage_partition(
    families: tuple[HoldoutFamily, ...] = CV_HOLDOUT,
) -> dict[str, str]:
    """The legacy LOFO grouping (by embryo lineage) — for side-by-side comparison."""
    return {f.name: f.lineage for f in families}


# --------------------------------------------------------------------------- #
# Parity statistics over an arbitrary holdout partition (pure).                #
# --------------------------------------------------------------------------- #


def group_micro_adj(
    rows: tuple[FamilyResult, ...] | list[FamilyResult],
    group_of: Callable[[str], str] | Mapping[str, str],
) -> dict[str, float]:
    """Per-group micro-adjusted edge Jaccard (reuses :func:`cv.aggregate`)."""
    of = group_of.get if isinstance(group_of, Mapping) else group_of
    buckets: dict[str, list[FamilyResult]] = {}
    for r in rows:
        buckets.setdefault(of(r.name), []).append(r)
    return {
        g: aggregate(grp).micro_adj_edge_jaccard for g, grp in sorted(buckets.items())
    }


def parity_micro_adj(
    rows: tuple[FamilyResult, ...] | list[FamilyResult],
    group_of: Callable[[str], str] | Mapping[str, str],
) -> float:
    """Mean over holdout groups of each group's micro-adjusted edge Jaccard.

    Parity, not sample size: every holdout unit (sequence / regime) counts
    equally, so a dense high-edge-count bucket cannot dominate the sparse one.
    """
    per = [v for v in group_micro_adj(rows, group_of).values() if not math.isnan(v)]
    return sum(per) / len(per) if per else float("nan")


# --------------------------------------------------------------------------- #
# Finer-grain transfer statistics per config + CV↔public transfer-trust.       #
# --------------------------------------------------------------------------- #


class HoldoutStats(NamedTuple):
    """Finer-grain holdout statistics for one historical config (all in [0, 1])."""

    name: str
    public_lb: float | None
    micro_adj: float  # live KPI (whole-holdout sample-weighted) — baseline
    lineage_macro_adj: float  # legacy LOFO parity (2 lineages)
    per_sequence_parity: float  # 4 embryo videos, parity (finest unit)
    regime_parity_adj: float  # 2 density regimes, parity (crosscuts lineage)
    sparse_regime_adj: float  # sparse-regime-only micro (single hardest stratum)
    dense_regime_adj: float  # dense-regime-only micro


def holdout_stats(
    config: ConfigCv,
    regime_by_family: Mapping[str, str] | None = None,
) -> HoldoutStats:
    """Compute every finer-grain holdout statistic for one config (pure)."""
    rows = tuple(config.rows)
    if regime_by_family is None:
        regime_by_family = regime_partition()
    cv = aggregate(rows)
    reg_micro = group_micro_adj(rows, regime_by_family)
    return HoldoutStats(
        name=config.name,
        public_lb=config.public_lb,
        micro_adj=cv.micro_adj_edge_jaccard,
        lineage_macro_adj=cv.lineage_macro_adj,
        per_sequence_parity=parity_micro_adj(rows, {r.name: r.name for r in rows}),
        regime_parity_adj=parity_micro_adj(rows, regime_by_family),
        sparse_regime_adj=reg_micro.get("sparse", float("nan")),
        dense_regime_adj=reg_micro.get("dense", float("nan")),
    )


def order_consistency(stats: list[HoldoutStats], statistic: str) -> dict:
    """Spearman of one finer-grain *statistic* vs public LB over anchored configs.

    Only configs carrying a same-metric ``public_lb`` are correlated (an
    unsubmitted / cross-patch config would mis-anchor the correlation). Reports
    the Spearman ρ and — the concrete transfer-trust check — whether the
    statistic *crowns the same config the public LB crowns*.
    """
    anchored = [s for s in stats if s.public_lb is not None]
    cv_vals = [getattr(s, statistic) for s in anchored]
    lb_vals = [s.public_lb for s in anchored]
    finite = [
        (c, lb)
        for c, lb in zip(cv_vals, lb_vals)
        if not (isinstance(c, float) and math.isnan(c))
    ]
    rho = spearman([c for c, _ in finite], [lb for _, lb in finite])
    top_public = max(anchored, key=lambda s: s.public_lb) if anchored else None
    top_cv = (
        max(
            (s for s in anchored if not math.isnan(getattr(s, statistic))),
            key=lambda s: getattr(s, statistic),
            default=None,
        )
        if anchored
        else None
    )
    return {
        "statistic": statistic,
        "n_anchored": len(anchored),
        "pairs": [
            {
                "name": s.name,
                "cv": (
                    None
                    if math.isnan(getattr(s, statistic))
                    else round(getattr(s, statistic), 4)
                ),
                "public_lb": s.public_lb,
            }
            for s in anchored
        ],
        "spearman_vs_public": (None if math.isnan(rho) else round(rho, 4)),
        "cv_top_matches_public_top": (
            None
            if top_public is None or top_cv is None
            else top_cv.name == top_public.name
        ),
    }


# Statistics ordered coarse → fine, so the transfer-trust table reads as a
# granularity sweep (whole holdout → lineage → sequence → regime → single regime).
TRANSFER_STATISTICS: tuple[str, ...] = (
    "micro_adj",
    "lineage_macro_adj",
    "per_sequence_parity",
    "regime_parity_adj",
    "sparse_regime_adj",
    "dense_regime_adj",
)


def leak_free_audit(
    regime_by_family: Mapping[str, str],
    families: tuple[HoldoutFamily, ...] = CV_HOLDOUT,
) -> dict:
    """Audit that the finer-grain partitions are leak-free (entity holdout kept).

    Confirms (a) the regime bucketing is a *strict partition* of the CV families
    (cover + disjoint — no whole embryo video split across buckets), and (b) the
    regime **crosscuts the lineage** (a lineage appears in more than one regime),
    which is the family-internal heterogeneity the lineage LOFO cannot express.
    """
    names = [f.name for f in families]
    labelled = {n: regime_by_family.get(n) for n in names}
    covered = all(labelled[n] not in (None, "unknown") for n in names)
    # Disjoint is automatic for a single-label mapping; assert every family maps
    # to exactly one regime and no extra names leak in.
    strict_partition = covered and set(regime_by_family) >= set(names)

    lineage_of = {f.name: f.lineage for f in families}
    regimes_per_lineage: dict[str, set[str]] = {}
    for n in names:
        regimes_per_lineage.setdefault(lineage_of[n], set()).add(labelled[n])
    crosscuts_lineage = any(len(v) > 1 for v in regimes_per_lineage.values())

    return {
        "entity_holdout_unit": "whole embryo video (.geff lineage)",
        "regime_by_family": dict(labelled),
        "strict_partition": strict_partition,
        "regimes_per_lineage": {k: sorted(v) for k, v in regimes_per_lineage.items()},
        "regime_crosscuts_lineage": crosscuts_lineage,
    }


def transfer_trust_report(
    lineage: list[ConfigCv] | tuple[ConfigCv, ...] = HISTORICAL_LINEAGE,
    *,
    regime_by_family: Mapping[str, str] | None = None,
    statistics: tuple[str, ...] = TRANSFER_STATISTICS,
    families: tuple[HoldoutFamily, ...] = CV_HOLDOUT,
) -> dict:
    """Full finer-grain CV↔public transfer-trust diagnosis over the lineage (pure).

    For each candidate holdout statistic (coarse whole-holdout micro → single
    density regime) it reports the Spearman ρ vs the same-metric public anchors
    and whether the statistic crowns the public winner. The accompanying
    leak-free audit shows the finer partitions never split an entity and the
    regime crosscuts the lineage. This is the evidence the SOT-2929 conclusion
    (finer granularity is leak-free but a strictly weaker private proxy) rests on.
    """
    if regime_by_family is None:
        regime_by_family = regime_partition()
    stats = [holdout_stats(c, regime_by_family) for c in lineage]
    table = [
        {
            "name": s.name,
            "public_lb": s.public_lb,
            **{
                stat: (
                    None if math.isnan(getattr(s, stat)) else round(getattr(s, stat), 4)
                )
                for stat in statistics
            },
        }
        for s in stats
    ]
    consistency = {stat: order_consistency(stats, stat) for stat in statistics}
    return {
        "regime_covariate": "median_knn_um",
        "regime_threshold_um": round(
            derive_regime_threshold(OBSERVED_MEDIAN_KNN_UM), 4
        ),
        "recommended_kpi": RECOMMENDED_KPI,
        "leak_free_audit": leak_free_audit(regime_by_family, families),
        "ranking_table": table,
        "order_consistency": consistency,
    }


def _main(argv: list[str] | None = None) -> int:
    """CLI: print the finer-grain transfer-trust diagnosis over the lineage."""
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Finer-grain leak-free holdout + CV↔public transfer-trust (SOT-2929)."
    )
    ap.add_argument("--out", type=str, default=None, help="Write the JSON report here.")
    args = ap.parse_args(argv)

    report = transfer_trust_report()
    print(json.dumps(report, indent=2))
    if args.out:
        from pathlib import Path

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
