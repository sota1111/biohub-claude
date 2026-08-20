"""Leak-free CV harness (SOT-2761): holdout integrity + pure aggregation.

The disk-touching entry point (:func:`evaluate_cv` over the real ``data/``
volumes) is verified in-session against the champion; here we cover everything
that does not need the (gitignored) competition data — the holdout definition,
the micro-averaging arithmetic, the no-regression gate, and the wiring of
``evaluate_cv`` with injected predictor + monkeypatched GT loaders.
"""

from __future__ import annotations

import math

from biohub_tracking.eval import cv as cv_mod
from biohub_tracking.eval.cv import (
    CV_HOLDOUT,
    SCORE_DIVISION_WEIGHT,
    FamilyResult,
    HoldoutFamily,
    aggregate,
    cv_result_to_dict,
    evaluate_cv,
    representativeness_report,
    score_family,
)
from biohub_tracking.eval.score import evaluate_datasets
from biohub_tracking.graph import TrackingGraph


def _line(n: int, offset: float = 0.0) -> TrackingGraph:
    """A simple linear track of *n* nodes, one per timepoint."""
    nodes = {i: (float(i), 0.0, offset, 0.0) for i in range(n)}
    edges = [(i, i + 1) for i in range(n - 1)]
    return TrackingGraph.from_lists(nodes, edges)


# --------------------------------------------------------------------------- #
# holdout definition — the leak-free entity split                             #
# --------------------------------------------------------------------------- #


def test_cv_holdout_is_the_four_real_test_families() -> None:
    names = [f.name for f in CV_HOLDOUT]
    assert names == ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
    # No family repeats — every scored entity is distinct.
    assert len(set(names)) == 4


def test_cv_holdout_entity_split_is_two_disjoint_lineages() -> None:
    lineages = {f.lineage for f in CV_HOLDOUT}
    assert lineages == {"44b6", "6bba"}
    # Balanced entity holdout: two embryo videos per lineage.
    for lin in lineages:
        assert sum(1 for f in CV_HOLDOUT if f.lineage == lin) == 2


def test_cv_holdout_scores_test_image_against_train_gt() -> None:
    # Leak-free mapping: scored input is the *test* volume, labels come from the
    # *train* GT of the same family — GT is used only to score, never to fit.
    for f in CV_HOLDOUT:
        assert f.image.startswith("data/test/") and f.image.endswith(".zarr")
        assert f.geff.startswith("data/train/") and f.geff.endswith(".geff")
        assert f.name in f.image and f.name in f.geff


# --------------------------------------------------------------------------- #
# pure aggregation — matches score.py micro-averaging                         #
# --------------------------------------------------------------------------- #


def test_score_family_counts_perfect_line() -> None:
    fam = HoldoutFamily("t", "44b6", "img", "gt")
    g = _line(5)
    fr = score_family(fam, g, g, n_true=5.0)
    assert (fr.edge_tp, fr.edge_fp, fr.edge_fn) == (4, 0, 0)
    assert fr.weight == 4
    assert fr.edge_jaccard == 1.0
    # 5 predicted nodes == n_true → no penalty, adjusted == raw.
    assert fr.adj_edge_jaccard == 1.0


def test_aggregate_micro_averages_and_splits_by_lineage() -> None:
    # Two lineages, each one family. Family A perfect (weight 4, adj 1.0);
    # family B half-right (tp=2, fp=2, fn=2 → jaccard 1/3, weight 6).
    a = _line(5)  # 4 edges
    fam_a = HoldoutFamily("a", "44b6", "i", "g")
    fr_a = score_family(fam_a, a, a, n_true=5.0)

    gt_b = _line(4, offset=0.0)  # 3 GT edges
    # Predict a disjoint extra track so we manufacture FPs/FNs deterministically.
    pred_b = TrackingGraph.from_lists(
        {0: (0.0, 0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0, 0.0),
         9: (0.0, 0.0, 100.0, 0.0), 10: (1.0, 0.0, 100.0, 0.0)},
        [(0, 1), (1, 2), (9, 10)],
    )
    fam_b = HoldoutFamily("b", "6bba", "i", "g")
    fr_b = score_family(fam_b, pred_b, gt_b, n_true=float("nan"))

    res = aggregate([fr_a, fr_b])

    # Micro edge jaccard = total_tp / (total_tp+fp+fn).
    total_tp = fr_a.edge_tp + fr_b.edge_tp
    total_fp = fr_a.edge_fp + fr_b.edge_fp
    total_fn = fr_a.edge_fn + fr_b.edge_fn
    assert math.isclose(
        res.micro_edge_jaccard, total_tp / (total_tp + total_fp + total_fn)
    )

    # Weight-averaged adjusted edge jaccard.
    expect_adj = (
        fr_a.weight * fr_a.adj_edge_jaccard + fr_b.weight * fr_b.adj_edge_jaccard
    ) / (fr_a.weight + fr_b.weight)
    assert math.isclose(res.micro_adj_edge_jaccard, expect_adj)

    # Per-lineage breakdown = each single-family adj (one family per lineage).
    assert math.isclose(res.by_lineage["44b6"], fr_a.adj_edge_jaccard)
    assert math.isclose(res.by_lineage["6bba"], fr_b.adj_edge_jaccard)

    # No GT divisions anywhere → division term dropped, score == micro adj.
    assert math.isnan(res.division_jaccard)
    assert res.score == res.micro_adj_edge_jaccard


def test_no_regression_gate() -> None:
    a = _line(5)
    fr = score_family(HoldoutFamily("a", "44b6", "i", "g"), a, a, n_true=5.0)
    res = aggregate([fr])
    assert res.no_regression_vs({"a": 0.9}) is True
    assert res.no_regression_vs({"a": 1.0}) is True  # tie is allowed
    assert res.no_regression_vs({"a": 1.1}) is False  # a real regression


def test_cv_result_to_dict_is_json_shaped() -> None:
    a = _line(5)
    fr = score_family(HoldoutFamily("a", "44b6", "i", "g"), a, a, n_true=5.0)
    d = cv_result_to_dict(aggregate([fr]))
    assert d["division_jaccard"] is None  # NaN → null
    assert d["per_dataset"][0]["name"] == "a"
    assert d["by_lineage"]["44b6"] == 1.0


# --------------------------------------------------------------------------- #
# evaluate_cv wiring — injected predictor + monkeypatched GT loaders          #
# --------------------------------------------------------------------------- #


def test_evaluate_cv_wires_end_to_end(monkeypatch) -> None:
    """evaluate_cv loads GT, runs the predictor per family, and aggregates.

    Monkeypatch the disk-backed loaders and the champion so this runs with no
    competition data; a perfect predictor (returns the GT) must yield micro 1.0.
    """
    gts = {f.name: _line(5) for f in CV_HOLDOUT}

    import biohub_tracking.champion as champ_mod
    import biohub_tracking.io as io_mod

    # GT keyed by family name (last path component's stem).
    def fake_load_geff(path):
        stem = str(path).split("/")[-1].replace(".geff", "")
        return gts[stem]

    monkeypatch.setattr(io_mod, "load_geff", fake_load_geff)
    monkeypatch.setattr(io_mod, "geff_scale", lambda p: (1.0, 1.0, 1.0))
    monkeypatch.setattr(io_mod, "geff_estimated_num_nodes", lambda p: 5.0)
    monkeypatch.setattr(champ_mod, "load_champion_config", lambda *a, **k: {"detect": {}, "link": {}})
    monkeypatch.setattr(
        champ_mod, "champion_params", lambda cfg=None: (object(), object(), (1.0, 1.0, 1.0))
    )

    # Perfect predictor: return the GT for the family being scored.
    def predictor(image, scale, detect, link):
        stem = str(image).split("/")[-1].replace(".zarr", "")
        return gts[stem]

    res = evaluate_cv(repo_root="/nonexistent", predictor=predictor)
    assert math.isclose(res.micro_adj_edge_jaccard, 1.0)
    assert res.by_lineage == {"44b6": 1.0, "6bba": 1.0}
    assert len(res.per_dataset) == 4


# --------------------------------------------------------------------------- #
# re-anchoring (SOT-2817): full competition metric + representativeness guard #
# --------------------------------------------------------------------------- #


def _division_pair() -> tuple[TrackingGraph, TrackingGraph]:
    """A graph with one recoverable division (grandparent→divider→children→..)."""
    nodes = {
        0: (0, 0, 0.0, 0), 1: (1, 0, 0.0, 0),
        2: (2, 0, 5.0, 0), 3: (2, 0, -5.0, 0),
        4: (3, 0, 5.0, 0), 5: (3, 0, -5.0, 0),
    }
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)]
    g = TrackingGraph.from_lists(nodes, edges)
    return g, g


def test_cv_arithmetic_matches_score_py() -> None:
    """The CV aggregation is byte-identical to eval/score.evaluate_datasets.

    Same (pred, gt) pairs + n_true through both paths must agree on the micro
    edge Jaccard, the weight-averaged adjusted edge Jaccard, the division
    Jaccard, and the combined score. This pins the oracle to the scorer so it
    cannot silently drift (the whole reason cv.py exists).
    """
    gt_a = _line(6)  # perfect, penalty-free
    div_g, _ = _division_pair()
    empty = TrackingGraph()
    pairs = [(gt_a, gt_a), (div_g, div_g), (empty, _line(4))]
    n_true = [6.0, 6.0, 4.0]

    ds = evaluate_datasets(pairs, n_true=n_true)

    fams = [HoldoutFamily(f"f{i}", "lin", "i", "g") for i in range(len(pairs))]
    rows = [
        score_family(fam, pred, gt, nt)
        for fam, (pred, gt), nt in zip(fams, pairs, n_true)
    ]
    cv = aggregate(rows)

    assert math.isclose(cv.micro_edge_jaccard, ds.edge_jaccard, rel_tol=1e-12)
    assert math.isclose(cv.micro_adj_edge_jaccard, ds.adj_edge_jaccard, rel_tol=1e-12)
    assert math.isclose(cv.division_jaccard, ds.division_jaccard, rel_tol=1e-12)
    assert math.isclose(cv.score, ds.score, rel_tol=1e-12)


def test_division_term_is_explicit_zero_when_no_fork_predicted() -> None:
    """A GT division that the prediction misses → term is 0.0, not null.

    This is the null-display fix: with the champion forfeiting divisions the
    confusion matrix is (tp=0, fp=0, fn>0), which IS measurable, so the division
    Jaccard is 0.0 and the term contributes an explicit 0.0 to the score.
    """
    div_g, _ = _division_pair()
    # Prediction is the linear backbone only — no fork → 1 division FN.
    pred = TrackingGraph.from_lists(
        {0: (0, 0, 0.0, 0), 1: (1, 0, 0.0, 0), 2: (2, 0, 5.0, 0), 4: (3, 0, 5.0, 0)},
        [(0, 1), (1, 2), (2, 4)],
    )
    fr = score_family(HoldoutFamily("d", "6bba", "i", "g"), pred, div_g, n_true=6.0)
    assert fr.division_fn >= 1 and fr.division_tp == 0
    res = aggregate([fr])
    assert res.division_measurable is True
    assert res.division_jaccard == 0.0
    assert res.division_term == 0.0
    assert math.isclose(res.score, res.micro_adj_edge_jaccard)


def test_division_term_zero_and_unmeasurable_without_gt_division() -> None:
    fr = score_family(HoldoutFamily("a", "44b6", "i", "g"), _line(5), _line(5), 5.0)
    res = aggregate([fr])
    assert res.division_measurable is False
    assert math.isnan(res.division_jaccard)
    assert res.division_term == 0.0  # explicit contribution, never dropped
    assert math.isclose(res.score, res.micro_adj_edge_jaccard)
    d = cv_result_to_dict(res)
    assert d["division_jaccard"] is None
    assert d["division_measurable"] is False
    assert d["division_term"] == 0.0


def test_division_term_enters_full_score_when_recovered() -> None:
    div_g, gt = _division_pair()
    fr = score_family(HoldoutFamily("d", "6bba", "i", "g"), div_g, gt, n_true=6.0)
    res = aggregate([fr])
    assert res.division_measurable is True
    assert res.division_jaccard == 1.0
    assert math.isclose(res.division_term, SCORE_DIVISION_WEIGHT * 1.0)
    assert math.isclose(
        res.score, res.micro_adj_edge_jaccard + SCORE_DIVISION_WEIGHT * 1.0
    )


def _champion_rows() -> list[FamilyResult]:
    """The frozen champion's per-family CV counts (SOT-2816 champion_cv_repro).

    Hard-coded so the re-anchored aggregation reproduces the registry 0.6649
    without the (gitignored) competition data — a pure regression anchor.
    """
    raw = [
        # name, lineage, tp, fp, fn, edge_j, adj, div(tp,fp,fn), pred, n_true, w
        ("44b6_0113de3b", "44b6", 47, 2, 3, 0.9038, 0.8895, (0, 0, 0), 29844, 25755.0, 52),
        ("44b6_0b24845f", "44b6", 37, 5, 12, 0.6852, 0.6817, (0, 0, 0), 34485, 32795.0, 54),
        ("6bba_05b6850b", "6bba", 651, 215, 194, 0.6142, 0.5700, (0, 0, 0), 10938, 6362.0, 1060),
        ("6bba_05db0fb1", "6bba", 973, 148, 210, 0.7310, 0.7310, (0, 0, 3), 69784, 69800.0, 1331),
    ]
    return [
        FamilyResult(
            name=n, lineage=lin, edge_tp=tp, edge_fp=fp, edge_fn=fn,
            edge_jaccard=ej, adj_edge_jaccard=adj,
            division_tp=d[0], division_fp=d[1], division_fn=d[2],
            num_pred_nodes=pred, n_true=nt, weight=w,
        )
        for (n, lin, tp, fp, fn, ej, adj, d, pred, nt, w) in raw
    ]


def test_reanchored_champion_reproduces_0_6649_with_explicit_division_zero() -> None:
    res = aggregate(_champion_rows())
    # Micro AND full score both 0.6649 — champion byte-invariant under re-anchor.
    assert round(res.micro_adj_edge_jaccard, 4) == 0.6649
    assert round(res.score, 4) == 0.6649
    # Division measurable (3 GT events) but forfeited → explicit 0.0 term.
    assert res.division_measurable is True
    assert res.division_jaccard == 0.0
    assert res.division_term == 0.0


def test_representativeness_guard_flags_family_mix_sensitivity() -> None:
    res = aggregate(_champion_rows())
    # Family-mix-robust views sit well above the 6bba-dominated micro.
    assert math.isclose(res.macro_adj_edge_jaccard, 0.71805, abs_tol=1e-4)
    assert round(res.lineage_macro_adj, 4) == 0.7216
    assert round(res.by_lineage_weight_share["6bba"], 4) == 0.9575
    rep = representativeness_report(res)
    assert rep["dominant_lineage"] == "6bba"
    assert rep["family_mix_sensitive"] is True  # gap 0.0567 > 0.05 tol
    assert rep["cv_public_order_consistent"] is True  # 0.6649 >= 0.624
    assert rep["cv_public_same_magnitude"] is True


def test_representativeness_report_detects_magnitude_and_order_breaks() -> None:
    # A CV a decimal order below the LB is neither order-consistent nor same-mag.
    a = _line(5)
    fr = score_family(HoldoutFamily("a", "44b6", "i", "g"), a, a, n_true=5.0)
    res = aggregate([fr])  # micro 1.0
    low = representativeness_report(res, public_lb_best=0.05)  # 1.0 / 0.05 = 20x
    assert low["cv_public_order_consistent"] is True
    assert low["cv_public_same_magnitude"] is False
    high = representativeness_report(res, public_lb_best=0.9)
    assert high["cv_public_order_consistent"] is True
    assert high["cv_public_same_magnitude"] is True
