"""CV↔public transfer diagnosis (SOT-2894) — pure, data-free coverage.

Verifies the re-anchoring logic that turns the SOT-2816 prose finding into a
gated statistic: the node-count-penalty decomposition, the lineage-parity
re-anchored KPI, the Spearman helper, and the order-consistency check against
the same-metric public anchors. None of this needs the (gitignored) data.
"""

from __future__ import annotations

import math

from biohub_tracking.eval.transfer import (
    HISTORICAL_LINEAGE,
    REANCHOR_GUARDRAIL,
    REANCHOR_PRIMARY,
    order_consistency,
    spearman,
    transfer_report,
    transfer_stats,
)


def _by_name(lineage):
    return {c.name: c for c in lineage}


# --------------------------------------------------------------------------- #
# Spearman helper                                                             #
# --------------------------------------------------------------------------- #


def test_spearman_perfect_and_inverse() -> None:
    assert math.isclose(spearman([1, 2, 3], [10, 20, 30]), 1.0)
    assert math.isclose(spearman([1, 2, 3], [30, 20, 10]), -1.0)


def test_spearman_handles_ties_and_degenerate() -> None:
    # One swap in the middle of 3 → ρ = 0.5 (the v1/v2 near-tie case).
    assert math.isclose(spearman([3, 1, 2], [3, 2, 1]), 0.5)
    # Zero rank variance → undefined, not a crash.
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert math.isnan(spearman([5.0], [1.0]))


# --------------------------------------------------------------------------- #
# node-count-penalty decomposition — the non-transferring layer               #
# --------------------------------------------------------------------------- #


def test_champion_reanchor_reproduces_registry_micro() -> None:
    champ = _by_name(HISTORICAL_LINEAGE)["detect-link-dog-v4-shorttrack"]
    s = transfer_stats(champ)
    # Legacy KPI still the byte-frozen registry 0.6649 (champion invariant).
    assert round(s.micro_adj, 4) == 0.6649
    # Penalty-stripped matching quality is a distinct, higher number.
    assert round(s.micro_raw, 4) == 0.684
    assert s.public_lb == 0.624


def test_node_penalty_is_the_dog_v2_to_v3_climb() -> None:
    """dog-v2→v3 micro_adj rises +0.10 while raw matching is flat.

    The whole adjusted gain is the node-count-penalty going from −0.134 to
    −0.030 (over-detection relief), NOT better matching — so a raw / re-anchored
    statistic must NOT credit v3 over v2. This is the attribution of the
    non-transferring CV climb to the node-count-penalty layer.
    """
    by = _by_name(HISTORICAL_LINEAGE)
    v2 = transfer_stats(by["detect-link-dog-v2"])
    v3 = transfer_stats(by["detect-link-dog-v3-adaptive"])

    # Legacy adjusted KPI: v3 "improves" ~+0.10.
    assert v3.micro_adj - v2.micro_adj > 0.09
    # Penalty-stripped matching quality: v3 does NOT improve (flat/slightly down).
    assert v3.micro_raw <= v2.micro_raw
    assert v3.lineage_macro_raw <= v2.lineage_macro_raw
    # The gain lives entirely in the penalty term (v2 heavily penalised).
    assert v2.node_penalty_contribution < -0.1
    assert v3.node_penalty_contribution > v2.node_penalty_contribution


# --------------------------------------------------------------------------- #
# order-consistency vs the same-metric public anchors                         #
# --------------------------------------------------------------------------- #


def test_only_same_metric_public_configs_are_anchored() -> None:
    stats = [transfer_stats(c) for c in HISTORICAL_LINEAGE]
    oc = order_consistency(stats, "micro_adj")
    # v3-adaptive has public_lb=None (unconfirmed / cross-patch) → excluded.
    assert oc["n_anchored"] == 3
    assert "detect-link-dog-v3-adaptive" not in [p["name"] for p in oc["pairs"]]


def test_champion_is_cv_top_and_public_top_no_negative_transfer() -> None:
    """The CV crowns the same config the pre-patch public LB crowns (champion).

    So there is NO cross-config negative transfer: Spearman vs public is
    positive (+0.5, the only discordance being the v1↔v2 0.509/0.500 near-tie),
    and the champion is the top on every candidate statistic. The apparent
    "CV up / public down" is the frozen champion's 0.624→0.509 metric-patch
    re-score, not a CV mis-ranking.
    """
    stats = [transfer_stats(c) for c in HISTORICAL_LINEAGE]
    for stat in ("micro_adj", "micro_raw", "lineage_macro_adj", "lineage_macro_raw"):
        oc = order_consistency(stats, stat)
        assert oc["spearman_vs_public"] == 0.5
        assert oc["cv_top_matches_public_top"] is True


def test_transfer_report_shape_and_reanchor_primary() -> None:
    report = transfer_report(list(HISTORICAL_LINEAGE))
    # SOT-2903: re-anchored to the OFFICIAL full metric (micro-adjusted).
    assert report["reanchor_primary_kpi"] == REANCHOR_PRIMARY == "micro_adj"
    assert len(report["ranking_table"]) == 4
    names = [r["name"] for r in report["ranking_table"]]
    assert names[0] == "detect-link-v1" and names[-1] == "detect-link-dog-v4-shorttrack"
    # Every anchored statistic is order-consistent (positive Spearman).
    for oc in report["order_consistency"].values():
        assert oc["spearman_vs_public"] is not None and oc["spearman_vs_public"] > 0


def test_sot2903_adjusted_beats_penalty_free_on_transfer_trust() -> None:
    """SOT-2903 audit: with the third public anchor (v3=0.557) the ADJUSTED
    (official-metric) statistics track public strictly better than the
    penalty-free ones — ρ=0.80 vs 0.40 — so the SOT-2894 penalty-free re-anchor
    was the weaker proxy and is corrected to ``micro_adj``.
    """
    # Add the v3-adaptive same-metric anchor (01c2f3=0.557, SOT-2300).
    lineage = [
        c._replace(public_lb=0.557) if c.name == "detect-link-dog-v3-adaptive" else c
        for c in HISTORICAL_LINEAGE
    ]
    stats = [transfer_stats(c) for c in lineage]
    adjusted = {
        stat: order_consistency(stats, stat)["spearman_vs_public"]
        for stat in ("micro_adj", "macro_adj", "lineage_macro_adj")
    }
    penalty_free = {
        stat: order_consistency(stats, stat)["spearman_vs_public"]
        for stat in ("micro_raw", "lineage_macro_raw")
    }
    assert all(round(v, 4) == 0.8 for v in adjusted.values()), adjusted
    assert all(round(v, 4) == 0.4 for v in penalty_free.values()), penalty_free
    # The re-anchored primary is an adjusted (true-metric) statistic, the
    # guardrail a raw one.
    assert REANCHOR_PRIMARY in adjusted and REANCHOR_GUARDRAIL in penalty_free
