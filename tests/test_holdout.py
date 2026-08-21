"""Finer-grain leak-free holdout + CV↔public transfer-trust (SOT-2929) — pure.

Covers the two deliverables direction B was opened for, all data-free:

1. a **finer-grain leak-free holdout** (per-sequence parity + observable
   density-regime stratification) with the leak-free invariants asserted
   (strict entity partition, regime derived from the observable covariate, regime
   crosscuts the lineage LOFO);
2. the **CV↔public transfer-trust** finding — the parity/blended statistics all
   rank the anchored public configs identically to the live KPI, while the
   *single-regime* statistic is a strictly weaker proxy (it fails to crown the
   public winner), so finer granularity does not improve the oracle.
"""

from __future__ import annotations

import math

from biohub_tracking.eval.cv import CV_HOLDOUT
from biohub_tracking.eval.holdout import (
    HISTORICAL_LINEAGE,
    OBSERVED_MEDIAN_KNN_UM,
    RECOMMENDED_KPI,
    by_lineage_partition,
    derive_regime_threshold,
    group_micro_adj,
    holdout_stats,
    leak_free_audit,
    order_consistency,
    parity_micro_adj,
    regime_partition,
    transfer_trust_report,
)


def _by_name(lineage):
    return {c.name: c for c in lineage}


# --------------------------------------------------------------------------- #
# Regime derivation from the observable GT-free covariate                     #
# --------------------------------------------------------------------------- #


def test_regime_threshold_isolates_the_sparse_tail() -> None:
    """The threshold is the high-side gap midpoint — between the dense cluster
    (≤ 8.54 µm) and the sparse outlier 6bba_05b6850b (9.49 µm), so ≈ 9.0 µm."""
    thr = derive_regime_threshold(OBSERVED_MEDIAN_KNN_UM)
    assert 8.6 < thr < 9.4


def test_regime_partition_reproduces_sot2921_observed_labels() -> None:
    """Labels come from the observable covariate (assign_regime), NOT the family
    prefix — and reproduce the SOT-2921 observed split: only the sparsest
    sequence is 'sparse'."""
    regime = regime_partition()
    assert regime == {
        "44b6_0113de3b": "dense",
        "44b6_0b24845f": "dense",
        "6bba_05b6850b": "sparse",
        "6bba_05db0fb1": "dense",
    }


# --------------------------------------------------------------------------- #
# Leak-free invariants of the finer-grain holdout                             #
# --------------------------------------------------------------------------- #


def test_finer_holdout_is_a_strict_entity_partition() -> None:
    """Every holdout unit is a whole embryo video and the regime bucketing is a
    strict partition of the CV families (cover + disjoint) — no entity split."""
    audit = leak_free_audit(regime_partition())
    assert audit["strict_partition"] is True
    assert audit["entity_holdout_unit"] == "whole embryo video (.geff lineage)"
    # Exactly the four CV families are labelled, each with one regime.
    assert set(audit["regime_by_family"]) == {f.name for f in CV_HOLDOUT}


def test_regime_crosscuts_the_lineage_lofo() -> None:
    """The density regime respects family-internal heterogeneity the lineage LOFO
    cannot: the 6bba lineage lands in BOTH regimes (the SOT-2921 crosscut)."""
    audit = leak_free_audit(regime_partition())
    assert audit["regime_crosscuts_lineage"] is True
    assert audit["regimes_per_lineage"]["6bba"] == ["dense", "sparse"]
    assert audit["regimes_per_lineage"]["44b6"] == ["dense"]


def test_per_sequence_parity_is_finer_than_lineage_parity() -> None:
    """Per-sequence parity weights all 4 embryo videos equally; lineage parity
    weights 2 lineages equally — a strictly finer holdout unit."""
    champ = _by_name(HISTORICAL_LINEAGE)["detect-link-dog-v4-shorttrack"]
    s = holdout_stats(champ)
    # 4 distinct singleton buckets vs 2 lineage buckets.
    per_seq = group_micro_adj(champ.rows, {r.name: r.name for r in champ.rows})
    assert len(per_seq) == 4
    assert len(by_lineage_partition()) == 4  # 4 families → 2 lineage labels
    assert len(set(by_lineage_partition().values())) == 2
    # per_sequence_parity == unweighted mean of the 4 per-family adjusted Jaccards.
    assert math.isclose(
        s.per_sequence_parity,
        sum(r.adj_edge_jaccard for r in champ.rows) / 4,
        rel_tol=1e-9,
    )


def test_parity_micro_adj_ignores_bucket_size() -> None:
    """A one-family bucket and a three-family bucket each count once."""
    champ = _by_name(HISTORICAL_LINEAGE)["detect-link-dog-v4-shorttrack"]
    reg = regime_partition()
    per_regime = group_micro_adj(champ.rows, reg)
    assert set(per_regime) == {"sparse", "dense"}
    assert math.isclose(
        parity_micro_adj(champ.rows, reg),
        (per_regime["sparse"] + per_regime["dense"]) / 2,
        rel_tol=1e-9,
    )


# --------------------------------------------------------------------------- #
# CV↔public transfer-trust — the direction-B finding                          #
# --------------------------------------------------------------------------- #


def test_champion_holdout_stats_reproduce_registry_micro() -> None:
    """Champion invariant: the whole-holdout micro is still the byte-frozen
    registry 0.6649; the sparse regime is a distinct, LOWER number (the hard
    stratum), the dense regime higher."""
    s = holdout_stats(_by_name(HISTORICAL_LINEAGE)["detect-link-dog-v4-shorttrack"])
    assert round(s.micro_adj, 4) == 0.6649
    assert s.sparse_regime_adj < s.micro_adj < s.dense_regime_adj


def test_parity_statistics_co_rank_public_like_the_live_kpi() -> None:
    """Every parity/blended statistic ranks the anchored public configs
    identically to the live KPI (same Spearman ρ) AND crowns the champion — so
    finer granularity adds no discriminating power over the current holdout."""
    stats = [holdout_stats(c) for c in HISTORICAL_LINEAGE]
    baseline = order_consistency(stats, "micro_adj")
    assert baseline["n_anchored"] == 3  # v3-adaptive public_lb=None → excluded
    assert baseline["cv_top_matches_public_top"] is True
    for stat in ("lineage_macro_adj", "per_sequence_parity", "regime_parity_adj"):
        oc = order_consistency(stats, stat)
        assert oc["spearman_vs_public"] == baseline["spearman_vs_public"]
        assert oc["cv_top_matches_public_top"] is True


def test_single_sparse_regime_is_a_strictly_weaker_proxy() -> None:
    """The sparse-regime-only statistic FAILS to crown the public winner: on the
    sparsest sequence (6bba_05b6850b) the champion is not the best config — the
    exact density-mix crosscut that broke SOT-2922/2923/2931. So a finer,
    single-regime holdout is a WORSE private proxy, not a better one."""
    stats = [holdout_stats(c) for c in HISTORICAL_LINEAGE]
    sparse = order_consistency(stats, "sparse_regime_adj")
    assert sparse["cv_top_matches_public_top"] is False
    # concretely: v1 beats the champion on the sparse sequence.
    by = {s.name: s for s in stats}
    assert (
        by["detect-link-v1"].sparse_regime_adj
        > by["detect-link-dog-v4-shorttrack"].sparse_regime_adj
    )


def test_transfer_trust_report_shape_and_conclusion() -> None:
    report = transfer_trust_report()
    assert report["recommended_kpi"] == RECOMMENDED_KPI == "micro_adj"
    assert report["leak_free_audit"]["regime_crosscuts_lineage"] is True
    assert len(report["ranking_table"]) == 4
    # The blended statistics are order-consistent; the single sparse regime is not.
    oc = report["order_consistency"]
    assert oc["micro_adj"]["cv_top_matches_public_top"] is True
    assert oc["regime_parity_adj"]["cv_top_matches_public_top"] is True
    assert oc["sparse_regime_adj"]["cv_top_matches_public_top"] is False
