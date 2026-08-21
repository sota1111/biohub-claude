"""GT-free density-regime covariate + leakage audit (SOT-2921).

These tests pin the leak-free guarantees the cycle-5 regime reformulation
depends on: the covariate uses **no ground truth**, is **deterministic**, runs on
**test-shape** inputs (no labels), does not depend on family identity, and the
regime-stratified CV report re-buckets the CV rows correctly. All of it runs on
synthetic point clouds so it needs none of the (gitignored) competition data.
"""

from __future__ import annotations

import math

import numpy as np

from biohub_tracking.detect import DEFAULT_SCALE
from biohub_tracking.eval.cv import FamilyResult
from biohub_tracking.eval.regime import (
    ACQUISITION_SCALE,
    assign_regime,
    regime_stratified_cv,
    separation_report,
    sequence_covariates,
)


def _lattice(spacing_um: float, n_per_axis: int = 6, n_frames: int = 5) -> dict:
    """A synthetic sequence: a cubic lattice of cells at a given physical spacing.

    Returned in **voxel** coordinates (the detector's native output), so the
    covariate must undo the anisotropic scale itself. Smaller ``spacing_um`` ⇒
    denser sequence.
    """
    scale = np.asarray(ACQUISITION_SCALE, dtype=float)
    step_vox = spacing_um / scale  # per-axis voxel step for the target micron spacing
    grid = np.arange(n_per_axis, dtype=float)
    zz, yy, xx = np.meshgrid(grid, grid, grid, indexing="ij")
    cloud_vox = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1) * step_vox
    # Same cloud every frame is enough to exercise the per-frame→median path.
    return {t: cloud_vox.copy() for t in range(n_frames)}


# --------------------------------------------------------------------------- #
# leakage audit: GT-free, deterministic, test-shape                           #
# --------------------------------------------------------------------------- #
def test_scale_constant_matches_detector():
    """ACQUISITION_SCALE must never drift from the detector's DEFAULT_SCALE."""
    assert ACQUISITION_SCALE == tuple(DEFAULT_SCALE)


def test_covariate_is_deterministic():
    """Same detections ⇒ byte-identical covariates (no RNG, no ordering effect)."""
    det = _lattice(spacing_um=10.0)
    a = sequence_covariates(det)
    b = sequence_covariates(det)
    assert a == b
    # Shuffling the point order within a frame must not change the result.
    shuffled = {t: pts[::-1].copy() for t, pts in det.items()}
    c = sequence_covariates(shuffled)
    assert math.isclose(a["median_knn_um"], c["median_knn_um"], rel_tol=1e-9)
    assert math.isclose(
        a["density_cells_per_1e6_um3"], c["density_cells_per_1e6_um3"], rel_tol=1e-9
    )


def test_covariate_signature_takes_only_detections():
    """The covariate depends only on the detection cloud — no GT can be passed.

    Guards against a future refactor sneaking a ground-truth graph in: the public
    entry point accepts a plain ``{t: (N,3)}`` mapping and a scale, nothing else
    that could carry labels.
    """
    import inspect

    params = list(inspect.signature(sequence_covariates).parameters)
    assert params[0] == "detections"
    # No parameter names anything like ground truth / labels / geff.
    assert not any(
        tok in p.lower() for p in params for tok in ("gt", "truth", "geff", "label")
    )


def test_covariate_runs_on_test_shape_input_without_labels():
    """Runs on a raw unlabeled point cloud of arbitrary shape (a 'test' input)."""
    det = {
        0: np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]),
        1: np.zeros((0, 3)),  # an empty frame must not crash
        2: np.array([[0.0, 0.0, 0.0]]),  # single cell < min → skipped for spacing
    }
    cov = sequence_covariates(det)
    assert cov["n_frames_total"] == 3
    assert cov["n_frames_scored"] == 1  # only frame 0 has >= 2 cells
    assert not math.isnan(cov["median_knn_um"])
    assert cov["n_detections_total"] == 4


def test_empty_sequence_yields_nan_not_crash():
    cov = sequence_covariates({0: np.zeros((0, 3)), 1: np.zeros((0, 3))})
    assert math.isnan(cov["median_knn_um"])
    assert cov["n_frames_scored"] == 0


# --------------------------------------------------------------------------- #
# the covariate actually measures density                                     #
# --------------------------------------------------------------------------- #
def test_spacing_recovers_physical_scale():
    """median_knn_um recovers the true physical lattice spacing (scale undone)."""
    cov = sequence_covariates(_lattice(spacing_um=12.0))
    assert math.isclose(cov["median_knn_um"], 12.0, rel_tol=1e-6)


def test_dense_sequence_has_smaller_spacing_and_higher_density():
    dense = sequence_covariates(_lattice(spacing_um=6.0))
    sparse = sequence_covariates(_lattice(spacing_um=18.0))
    assert dense["median_knn_um"] < sparse["median_knn_um"]
    assert (
        dense["density_cells_per_1e6_um3"] > sparse["density_cells_per_1e6_um3"]
    )
    assert dense["local_crowding"] >= sparse["local_crowding"]


# --------------------------------------------------------------------------- #
# separation report                                                           #
# --------------------------------------------------------------------------- #
def _family_of(name: str) -> str:
    return name.split("_", 1)[0]


def test_separation_report_detects_perfect_split():
    covs = {
        "6bba_a": sequence_covariates(_lattice(spacing_um=6.0)),
        "6bba_b": sequence_covariates(_lattice(spacing_um=7.0)),
        "44b6_a": sequence_covariates(_lattice(spacing_um=16.0)),
        "44b6_b": sequence_covariates(_lattice(spacing_um=18.0)),
    }
    rep = separation_report(covs, _family_of, key="median_knn_um", dense_is_low=True)
    assert rep["separable"] is True
    assert rep["margin"] > 0
    assert rep["dense_group"] == "6bba"
    assert rep["all_correct"] is True
    # Threshold sits between the two groups.
    assert 7.0 < rep["threshold"] < 16.0


def test_separation_report_flags_overlap():
    covs = {
        "6bba_a": sequence_covariates(_lattice(spacing_um=10.0)),
        "44b6_a": sequence_covariates(_lattice(spacing_um=11.0)),
        "6bba_b": sequence_covariates(_lattice(spacing_um=12.0)),  # overlaps 44b6 side
        "44b6_b": sequence_covariates(_lattice(spacing_um=9.0)),
    }
    rep = separation_report(covs, _family_of, key="median_knn_um", dense_is_low=True)
    assert rep["separable"] is False
    assert rep["margin"] <= 0


def test_assign_regime_orientation():
    # spacing covariate: low ⇒ dense
    assert assign_regime(5.0, 10.0, dense_is_low=True) == "dense"
    assert assign_regime(15.0, 10.0, dense_is_low=True) == "sparse"
    # count covariate: high ⇒ dense
    assert assign_regime(500.0, 100.0, dense_is_low=False) == "dense"
    assert assign_regime(50.0, 100.0, dense_is_low=False) == "sparse"
    assert assign_regime(float("nan"), 10.0) == "unknown"


# --------------------------------------------------------------------------- #
# regime-stratified CV report                                                 #
# --------------------------------------------------------------------------- #
def _fam(name: str, lineage: str, tp: int, fp: int, fn: int, adj: float) -> FamilyResult:
    j = tp / (tp + fp + fn)
    return FamilyResult(
        name=name,
        lineage=lineage,
        edge_tp=tp,
        edge_fp=fp,
        edge_fn=fn,
        edge_jaccard=j,
        adj_edge_jaccard=adj,
        division_tp=0,
        division_fp=0,
        division_fn=0,
        num_pred_nodes=tp + fp,
        n_true=float(tp + fn),
        weight=tp + fp + fn,
    )


def test_regime_stratified_cv_buckets_by_regime():
    rows = [
        _fam("44b6_0113de3b", "44b6", 90, 5, 5, 0.90),
        _fam("44b6_0b24845f", "44b6", 70, 15, 15, 0.70),
        _fam("6bba_05b6850b", "6bba", 570, 200, 230, 0.57),
        _fam("6bba_05db0fb1", "6bba", 730, 130, 140, 0.75),
    ]
    regime = {
        "44b6_0113de3b": "sparse",
        "44b6_0b24845f": "sparse",
        "6bba_05b6850b": "dense",
        "6bba_05db0fb1": "dense",
    }
    rep = regime_stratified_cv(rows, regime)
    assert set(rep["per_regime"]) == {"sparse", "dense"}
    assert rep["per_regime"]["sparse"]["families"] == [
        "44b6_0113de3b",
        "44b6_0b24845f",
    ]
    # Dense regime carries the vast majority of the edge weight (family-mix wall).
    assert rep["per_regime"]["dense"]["weight_share"] > 0.9
    # Per-regime micro is the sample-weighted mean within the bucket.
    sparse = rep["per_regime"]["sparse"]
    assert math.isclose(sparse["micro_edge_jaccard"], 160 / (160 + 20 + 20), rel_tol=1e-6)
    assert rep["per_family"]["6bba_05b6850b"]["regime"] == "dense"
