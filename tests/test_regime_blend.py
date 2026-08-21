"""Soft per-sequence operating-point mixture (SOT-2931, default-off).

These tests pin the thin soft-mixture layer that selects a **continuously
blended** linking operating point per sequence from the observable density
covariate — the soft-mixture reformulation of the SOT-2922 hard regime label:

* **default-off** — the champion config carries no ``operating_point_mixture``
  block, so :func:`biohub_tracking.champion.operating_point_mixture_policy`
  returns ``None`` and ``champion_params`` is byte-for-byte unchanged;
* **continuity** — the blended ``motion_gain`` moves *continuously* with the
  covariate (not a step), and the hard-switch limit (``scale → 0``) reproduces the
  SOT-2922 two-way partition, so the soft policy strictly generalises the hard one;
* **champion safety** — an unclassifiable (NaN) covariate and the degenerate
  ``center = ±inf`` policy both fall back to the champion endpoint;
* **leak-free fit** — :func:`fit_fold_policy` fits center/scale/gate/gain on
  *training* families only, honours the non-regression gate, and falls back to the
  champion when no soft policy clears it.

All of it runs on synthetic ``FamilyResult`` rows, so it needs none of the
(gitignored) competition data.
"""

from __future__ import annotations

from biohub_tracking.champion import (
    EMBEDDED_CHAMPION_CONFIG,
    champion_params,
    load_champion_config,
    operating_point_mixture_policy,
)
from biohub_tracking.eval.cv import FamilyResult
from biohub_tracking.eval.regime_blend import (
    CHAMPION_ENDPOINT,
    BlendEndpoint,
    SoftBlendPolicy,
    center_candidates,
    fit_fold_policy,
    op_key,
    snap_gain,
)

GAIN_GRID = [1.0, 1.25, 1.5, 1.75, 2.0]
AGGR = BlendEndpoint(motion_gain=2.0, cycle_consistency_gate=True, cycle_consistency_margin=0.0)


def _row(name: str, adj: float, *, weight: int = 1000, raw: float | None = None) -> FamilyResult:
    raw = adj if raw is None else raw
    return FamilyResult(
        name=name,
        lineage=name.split("_")[0],
        edge_tp=int(adj * weight),
        edge_fp=0,
        edge_fn=weight - int(adj * weight),
        edge_jaccard=raw,
        adj_edge_jaccard=adj,
        division_tp=0,
        division_fp=0,
        division_fn=0,
        num_pred_nodes=weight,
        n_true=float(weight),
        weight=weight,
    )


# ---------------------------------------------------------------- default-off ---
def test_champion_config_has_no_mixture_block():
    cfg = load_champion_config()
    assert operating_point_mixture_policy(cfg) is None
    # Embedded champion is byte-clean too.
    assert operating_point_mixture_policy(dict(EMBEDDED_CHAMPION_CONFIG)) is None


def test_champion_params_unchanged_by_absent_block():
    _detect, link, _scale = champion_params(load_champion_config())
    # The champion linking op the mixture would blend around.
    assert link.motion_gain == 1.0
    assert link.cycle_consistency_gate is False


def test_policy_reader_roundtrips_a_present_block():
    block = SoftBlendPolicy(
        covariate_key="median_knn_um",
        center=9.0,
        scale=0.5,
        dense_is_low=True,
        conservative=CHAMPION_ENDPOINT,
        aggressive=AGGR,
        gate_activation=0.5,
    ).to_dict()
    cfg = {"operating_point_mixture": block}
    pol = operating_point_mixture_policy(cfg)
    assert isinstance(pol, SoftBlendPolicy)
    assert pol.center == 9.0 and pol.scale == 0.5
    # Also accepted nested under ``link``.
    assert operating_point_mixture_policy({"link": {"operating_point_mixture": block}}) is not None


# ---------------------------------------------------------------- continuity ----
def test_weight_is_continuous_and_monotone():
    pol = SoftBlendPolicy(
        covariate_key="median_knn_um",
        center=8.5,
        scale=0.5,
        dense_is_low=True,
        conservative=CHAMPION_ENDPOINT,
        aggressive=AGGR,
        gate_activation=0.5,
    )
    # dense_is_low: smaller covariate (denser) => larger aggressive weight.
    xs = [7.0, 8.0, 8.5, 9.0, 10.0]
    ws = [pol.weight_of(x) for x in xs]
    assert all(0.0 < w < 1.0 for w in ws)
    assert ws == sorted(ws, reverse=True)  # strictly decreasing in x
    assert abs(pol.weight_of(8.5) - 0.5) < 1e-9  # w = 0.5 exactly at center
    # Blended gain slides continuously between the endpoints.
    gains = [pol.op_for(x).motion_gain for x in xs]
    assert all(1.0 < g < 2.0 for g in gains)
    assert gains == sorted(gains, reverse=True)


def test_hard_switch_limit_reproduces_two_way_partition():
    # scale -> 0 is the SOT-2922 hard step at ``center``.
    pol = SoftBlendPolicy(
        covariate_key="median_knn_um",
        center=8.5,
        scale=0.0,
        dense_is_low=True,
        conservative=CHAMPION_ENDPOINT,
        aggressive=AGGR,
        gate_activation=0.5,
    )
    # x below center => dense => aggressive weight 1.0 => aggressive endpoint.
    assert pol.weight_of(8.0) == 1.0
    assert pol.op_for(8.0).motion_gain == 2.0
    assert pol.op_for(8.0).cycle_consistency_gate is True
    # x above center => sparse => conservative (champion) endpoint.
    assert pol.weight_of(9.0) == 0.0
    assert pol.op_for(9.0).motion_gain == 1.0
    assert pol.op_for(9.0).cycle_consistency_gate is False


def test_gate_activation_thresholds_prune_on_continuous_weight():
    base = {
        "covariate_key": "median_knn_um",
        "center": 8.5,
        "scale": 0.5,
        "dense_is_low": True,
        "conservative": CHAMPION_ENDPOINT,
        "aggressive": AGGR,
    }
    never = SoftBlendPolicy(**base, gate_activation=float("inf"))
    always = SoftBlendPolicy(**base, gate_activation=0.0)
    # Pure gain blend: prune never engages regardless of weight.
    assert never.op_for(7.0).cycle_consistency_gate is False
    # Aggressive-weight side engages the prune once w >= activation.
    assert always.op_for(7.0).cycle_consistency_gate is True


# ------------------------------------------------------------- champion safety --
def test_nan_and_degenerate_center_fall_back_to_champion():
    pol = SoftBlendPolicy(
        covariate_key="median_knn_um",
        center=8.5,
        scale=0.5,
        dense_is_low=True,
        conservative=CHAMPION_ENDPOINT,
        aggressive=AGGR,
        gate_activation=0.5,
    )
    nan_op = pol.op_for(float("nan"))
    assert nan_op.motion_gain == 1.0 and nan_op.cycle_consistency_gate is False
    # center = -inf (dense_is_low) => nothing dense => all conservative.
    allcons = SoftBlendPolicy(
        covariate_key="median_knn_um",
        center=float("-inf"),
        scale=0.5,
        dense_is_low=True,
        conservative=CHAMPION_ENDPOINT,
        aggressive=AGGR,
        gate_activation=0.5,
    )
    assert allcons.op_for(8.0).motion_gain == 1.0


def test_snap_gain_and_op_key():
    assert snap_gain(1.6, GAIN_GRID) == 1.5
    assert snap_gain(1.7, GAIN_GRID) == 1.75
    # Tie-break: nearest, lower wins.
    assert snap_gain(1.125, GAIN_GRID) == 1.0
    assert op_key(CHAMPION_ENDPOINT, GAIN_GRID) == (1.0, False)
    assert op_key(AGGR, GAIN_GRID) == (2.0, True)


# --------------------------------------------------------------- leak-free fit --
def _scored_grid(adj_by_fam_op):
    """Build a ``scored[(fam,(gain,gate))]`` map from a nested dict of adj values."""
    scored = {}
    for fam, ops in adj_by_fam_op.items():
        for op, adj in ops.items():
            scored[(fam, op)] = _row(fam, adj)
    return scored


def test_fit_prefers_soft_policy_that_clears_training_gate():
    # Three training families; the aggressive op helps the dense one and does not
    # regress the others => a soft mixture should be selected over champion.
    champ = {"a": 0.60, "b": 0.70, "c": 0.65}
    covariate = {"a": 7.0, "b": 8.5, "c": 9.5}  # a densest
    scored = _scored_grid(
        {  # full gain × gate cross product the fit may land on
            "a": {(1.0, False): 0.60, (1.0, True): 0.58, (2.0, False): 0.62, (2.0, True): 0.66},
            "b": {(1.0, False): 0.70, (1.0, True): 0.68, (2.0, False): 0.70, (2.0, True): 0.70},
            "c": {(1.0, False): 0.65, (1.0, True): 0.60, (2.0, False): 0.63, (2.0, True): 0.62},
        }
    )
    fit = fit_fold_policy(
        ["a", "b", "c"],
        covariate,
        scored,
        gain_grid=[1.0, 2.0],
        scale_grid=[0.5],
        gate_activation_grid=[0.5],
        aggressive_gain_grid=[2.0],
        champion_adj_by_family=champ,
        covariate_key="median_knn_um",
        dense_is_low=True,
    )
    assert fit.train_no_regression is True
    assert not fit.fell_back_to_champion
    # a lands on the aggressive op; c stays champion.
    assert op_key(fit.policy.op_for(covariate["a"]), [1.0, 2.0]) == (2.0, True)
    assert op_key(fit.policy.op_for(covariate["c"]), [1.0, 2.0]) == (1.0, False)


def test_fit_falls_back_to_champion_when_no_policy_clears_gate():
    # Aggressive op regresses everyone => only champion-everywhere is non-regressing.
    champ = {"a": 0.60, "b": 0.70, "c": 0.65}
    covariate = {"a": 7.0, "b": 8.5, "c": 9.5}
    scored = _scored_grid(
        {
            "a": {(1.0, False): 0.60, (1.0, True): 0.55, (2.0, False): 0.56, (2.0, True): 0.55},
            "b": {(1.0, False): 0.70, (1.0, True): 0.62, (2.0, False): 0.61, (2.0, True): 0.60},
            "c": {(1.0, False): 0.65, (1.0, True): 0.52, (2.0, False): 0.55, (2.0, True): 0.50},
        }
    )
    fit = fit_fold_policy(
        ["a", "b", "c"],
        covariate,
        scored,
        gain_grid=[1.0, 2.0],
        scale_grid=[0.5],
        gate_activation_grid=[0.5],
        aggressive_gain_grid=[2.0],
        champion_adj_by_family=champ,
        covariate_key="median_knn_um",
        dense_is_low=True,
    )
    assert fit.fell_back_to_champion is True
    assert fit.train_no_regression is True
    for f in ["a", "b", "c"]:
        assert op_key(fit.policy.op_for(covariate[f]), [1.0, 2.0]) == (1.0, False)


def test_center_candidates_are_training_only_midpoints():
    cands = center_candidates([9.5, 7.0, 8.5])
    assert cands[0] == float("-inf") and cands[-1] == float("inf")
    mids = cands[1:-1]
    assert mids == [7.75, 9.0]  # midpoints of sorted [7.0,8.5,9.5]
    # NaN dropped.
    assert center_candidates([7.0, float("nan"), 9.0])[1:-1] == [8.0]
