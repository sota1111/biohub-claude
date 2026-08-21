"""Regime-conditional *linking* operating point (SOT-2922, default-off).

These tests pin the thin conditional layer that selects the linking levers
(``motion_gain`` + mutual-NN ``cycle_consistency_gate``/``margin``) per sequence
from the observable density covariate:

* **default-off** — the champion config carries no ``regime_conditional_link``
  block, so :func:`biohub_tracking.champion.regime_conditional_link_policy`
  returns ``None`` and ``champion_params`` is byte-for-byte unchanged;
* **policy application** — a present block maps dense/sparse covariate values to
  the right linking operating point (and an unknown/NaN covariate falls back to
  the champion / recall-preserving sparse side);
* **leak-free fit** — :func:`fit_fold_policy` fits threshold + per-regime
  operating points on *training* families only, honours the non-regression gate,
  and falls back to the champion operating point when no split clears it.

All of it runs on synthetic ``FamilyResult`` rows, so it needs none of the
(gitignored) competition data.
"""

from __future__ import annotations

import math

from biohub_tracking.champion import (
    EMBEDDED_CHAMPION_CONFIG,
    champion_params,
    load_champion_config,
    regime_conditional_link_policy,
)
from biohub_tracking.eval.cv import FamilyResult
from biohub_tracking.eval.regime_link import (
    ConditionalLinkPolicy,
    RegimeLinkPoint,
    fit_fold_policy,
    threshold_candidates,
)

# Champion linking op: motion_gain=1.0, mutual-NN prune OFF.
CHAMP_OP = RegimeLinkPoint(motion_gain=1.0, cycle_consistency_gate=False)
# Aggressive dense op: motion_gain=2.0 (SOT-2900) + mutual-NN prune (SOT-2910).
AGGR_OP = RegimeLinkPoint(motion_gain=2.0, cycle_consistency_gate=True, cycle_consistency_margin=0.0)


def _row(name: str, adj: float, weight: int = 1000, lineage: str = "x") -> FamilyResult:
    """A minimal FamilyResult carrying only what the policy layer reads."""
    return FamilyResult(
        name=name,
        lineage=lineage,
        edge_tp=weight,
        edge_fp=0,
        edge_fn=0,
        edge_jaccard=adj,
        adj_edge_jaccard=adj,
        division_tp=0,
        division_fp=0,
        division_fn=0,
        num_pred_nodes=weight,
        n_true=float(weight),
        weight=weight,
    )


# --------------------------------------------------------------------------- #
# default-off: champion carries no block ⇒ None ⇒ byte-identical params        #
# --------------------------------------------------------------------------- #
def test_champion_config_has_no_regime_conditional_link_block():
    assert "regime_conditional_link" not in EMBEDDED_CHAMPION_CONFIG
    assert "regime_conditional_link" not in EMBEDDED_CHAMPION_CONFIG["link"]


def test_regime_conditional_link_policy_is_none_for_champion():
    assert regime_conditional_link_policy(load_champion_config()) is None
    assert regime_conditional_link_policy(EMBEDDED_CHAMPION_CONFIG) is None


def test_champion_params_unchanged_when_flag_off():
    # The champion linking knobs are exactly the frozen values; the default-off
    # layer never mutates them.
    _detect, link, _scale = champion_params(EMBEDDED_CHAMPION_CONFIG)
    assert link.motion_gain == 1.0
    assert link.cycle_consistency_gate is False


# --------------------------------------------------------------------------- #
# policy round-trip + application                                             #
# --------------------------------------------------------------------------- #
def test_policy_from_dict_selects_op_by_regime():
    block = {
        "covariate_key": "median_knn_um",
        "threshold": 8.0,
        "dense_is_low": True,
        "dense_op": {"motion_gain": 2.0, "cycle_consistency_gate": True},
        "sparse_op": {"motion_gain": 1.0, "cycle_consistency_gate": False},
    }
    policy = ConditionalLinkPolicy.from_dict(block)
    # dense_is_low: value <= threshold ⇒ dense ⇒ aggressive op.
    assert policy.regime_of(7.0) == "dense"
    assert policy.op_for(7.0) == RegimeLinkPoint(2.0, True, 0.0)
    # value > threshold ⇒ sparse ⇒ champion op.
    assert policy.regime_of(9.0) == "sparse"
    assert policy.op_for(9.0) == RegimeLinkPoint(1.0, False, 0.0)


def test_policy_present_block_reads_back_via_config_reader():
    cfg = {
        **EMBEDDED_CHAMPION_CONFIG,
        "regime_conditional_link": {
            "threshold": 8.0,
            "dense_op": {"motion_gain": 2.0, "cycle_consistency_gate": True},
            "sparse_op": {"motion_gain": 1.0, "cycle_consistency_gate": False},
        },
    }
    policy = regime_conditional_link_policy(cfg)
    assert isinstance(policy, ConditionalLinkPolicy)
    assert policy.op_for(7.0).motion_gain == 2.0
    assert policy.op_for(7.0).cycle_consistency_gate is True


def test_unknown_covariate_falls_back_to_sparse_side():
    policy = ConditionalLinkPolicy(
        covariate_key="median_knn_um",
        threshold=8.0,
        dense_is_low=True,
        dense_op=AGGR_OP,
        sparse_op=CHAMP_OP,
    )
    # NaN ⇒ "unknown" ⇒ champion side, never the aggressive dense op.
    assert policy.regime_of(float("nan")) == "unknown"
    assert policy.op_for(float("nan")) == CHAMP_OP


def test_to_dict_from_dict_round_trip():
    policy = ConditionalLinkPolicy(
        covariate_key="median_knn_um",
        threshold=8.5,
        dense_is_low=True,
        dense_op=AGGR_OP,
        sparse_op=CHAMP_OP,
    )
    assert ConditionalLinkPolicy.from_dict(policy.to_dict()) == policy


# --------------------------------------------------------------------------- #
# threshold candidates: leak-free splits from training values only            #
# --------------------------------------------------------------------------- #
def test_threshold_candidates_are_training_midpoints_plus_edges():
    cands = threshold_candidates([9.0, 7.0, 8.0])
    assert cands[0] == float("-inf")
    assert cands[-1] == float("inf")
    assert 7.5 in cands and 8.5 in cands  # midpoints of sorted 7,8,9


def test_threshold_candidates_drop_nan():
    cands = threshold_candidates([8.0, float("nan"), 9.0])
    assert 8.5 in cands
    assert all(not math.isnan(c) for c in cands if math.isfinite(c))


# --------------------------------------------------------------------------- #
# leak-free fit: training only, non-regression gate, champion fallback         #
# --------------------------------------------------------------------------- #
def test_fit_picks_regime_split_when_it_helps_without_regressing():
    # 3 training families. dense_a (cov 7.0) gains a lot under AGGR_OP; the two
    # sparse families (cov 9.0/9.5) are hurt by AGGR_OP but fine under CHAMP_OP.
    op_grid = [CHAMP_OP, AGGR_OP]
    cov = {"dense_a": 7.0, "sparse_b": 9.0, "sparse_c": 9.5}
    scored = {
        ("dense_a", CHAMP_OP): _row("dense_a", 0.60),
        ("dense_a", AGGR_OP): _row("dense_a", 0.75),  # dense loves aggressive op
        ("sparse_b", CHAMP_OP): _row("sparse_b", 0.70),
        ("sparse_b", AGGR_OP): _row("sparse_b", 0.50),  # aggressive hurts sparse
        ("sparse_c", CHAMP_OP): _row("sparse_c", 0.80),
        ("sparse_c", AGGR_OP): _row("sparse_c", 0.55),
    }
    champ_adj = {"dense_a": 0.60, "sparse_b": 0.70, "sparse_c": 0.80}
    fit = fit_fold_policy(
        ["dense_a", "sparse_b", "sparse_c"],
        cov,
        scored,
        op_grid=op_grid,
        champion_op=CHAMP_OP,
        champion_adj_by_family=champ_adj,
    )
    # The fitted policy must give the dense family AGGR_OP and the sparse ones
    # CHAMP_OP, so no training family regresses and the dense one gains.
    assert fit.train_no_regression
    assert not fit.fell_back_to_champion
    assert fit.policy.op_for(7.0) == AGGR_OP
    assert fit.policy.op_for(9.0) == CHAMP_OP
    assert fit.policy.op_for(9.5) == CHAMP_OP
    # 7.0 dense, 9.0/9.5 sparse ⇒ threshold between 7 and 9.
    assert 7.0 < fit.policy.threshold <= 9.0


def test_fit_falls_back_to_champion_when_no_split_clears_gate():
    # AGGR_OP regresses every family and no split avoids that ⇒ champion op only.
    op_grid = [CHAMP_OP, AGGR_OP]
    cov = {"a": 7.0, "b": 8.0, "c": 9.0}
    scored = {
        ("a", CHAMP_OP): _row("a", 0.70),
        ("a", AGGR_OP): _row("a", 0.60),
        ("b", CHAMP_OP): _row("b", 0.70),
        ("b", AGGR_OP): _row("b", 0.60),
        ("c", CHAMP_OP): _row("c", 0.70),
        ("c", AGGR_OP): _row("c", 0.60),
    }
    champ_adj = {"a": 0.70, "b": 0.70, "c": 0.70}
    fit = fit_fold_policy(
        ["a", "b", "c"], cov, scored,
        op_grid=op_grid, champion_op=CHAMP_OP, champion_adj_by_family=champ_adj,
    )
    assert fit.fell_back_to_champion
    assert fit.policy.op_for(7.0) == CHAMP_OP
    assert fit.policy.op_for(9.0) == CHAMP_OP


def test_fit_only_reads_training_families():
    # A held-out family present in cov/scored but NOT in train_families must not
    # affect the fit (leak-free): its score is deliberately absurd.
    op_grid = [CHAMP_OP, AGGR_OP]
    cov = {"a": 7.0, "b": 9.0, "HELDOUT": 7.0}
    scored = {
        ("a", CHAMP_OP): _row("a", 0.60),
        ("a", AGGR_OP): _row("a", 0.75),
        ("b", CHAMP_OP): _row("b", 0.80),
        ("b", AGGR_OP): _row("b", 0.55),
        # Held-out rows exist but must be ignored by the fit.
        ("HELDOUT", CHAMP_OP): _row("HELDOUT", 0.01),
        ("HELDOUT", AGGR_OP): _row("HELDOUT", 0.99),
    }
    champ_adj = {"a": 0.60, "b": 0.80, "HELDOUT": 0.01}
    fit = fit_fold_policy(
        ["a", "b"], cov, scored,
        op_grid=op_grid, champion_op=CHAMP_OP, champion_adj_by_family=champ_adj,
    )
    # dense a ⇒ AGGR, sparse b ⇒ CHAMP; the absurd HELDOUT row never leaks in.
    assert fit.policy.op_for(7.0) == AGGR_OP
    assert fit.policy.op_for(9.0) == CHAMP_OP
