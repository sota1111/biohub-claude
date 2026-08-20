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
    HoldoutFamily,
    aggregate,
    cv_result_to_dict,
    evaluate_cv,
    score_family,
)
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
