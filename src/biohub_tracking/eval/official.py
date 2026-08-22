"""Official royerlab scorer bridge — the oracle-fidelity anchor (SOT-2995).

The rest of :mod:`biohub_tracking.eval` is a **clean-room reimplementation** of
the competition metric (numpy + scipy only, so it runs unchanged inside a Kaggle
submission kernel). A clean-room oracle is only trustworthy if it actually agrees
with the organiser's own scorer — otherwise a silent divergence between the two
IS the generalization-gap culprit for every learning experiment that trusts the
local CV.

This module removes the "assume equal" step: it runs the **genuine** organiser
code —
``royerlab/kaggle-cell-tracking-competition`` (``tracking_cellmot.metrics`` +
``tracking_cellmot.division_metrics``, which use ``tracksdata`` + ``polars``) — on
the *same* :class:`~biohub_tracking.graph.TrackingGraph` predictions the
clean-room scorer sees, and reports the per-count divergence.

Portability guarantee
----------------------
``tracksdata`` / ``tracking_cellmot`` are heavy, torch-adjacent, dev-only
dependencies. They are **never** imported at module load — every entry point
imports them lazily and raises :class:`OfficialScorerUnavailable` with install
instructions if they are missing. Importing this module (or the wider
``biohub_tracking.eval`` package, which does not import it) therefore never pulls
them into the light kernel path, and the golden CI tests stay green without them.

The bridge builds an in-memory ``tracksdata`` graph node-for-node / edge-for-edge
from a :class:`TrackingGraph` (no disk round-trip), then calls the official
``evaluate`` / ``per_sample_metrics`` / ``summarise`` verbatim. The returned
counts use the SAME :class:`biohub_tracking.eval.score.EvaluationResult` /
:class:`biohub_tracking.eval.cv.FamilyResult` shapes as the clean-room path, so a
caller can diff them directly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

from ..graph import TrackingGraph
from ..matching import DEFAULT_MAX_DISTANCE
from .cv import FamilyResult
from .score import ADJUSTMENT_ALPHA, SCORE_DIVISION_WEIGHT, EvaluationResult, _jaccard

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


class OfficialScorerUnavailable(RuntimeError):
    """Raised when the official ``tracksdata`` / ``tracking_cellmot`` stack is absent."""


_INSTALL_HINT = (
    "The official royerlab scorer requires `tracksdata` and `tracking_cellmot` "
    "(heavy dev-only deps). Install into a SEPARATE venv (never the kernel path):\n"
    "  pip install polars scipy numpy rustworkx 'zarr>=3' geff \\\n"
    "      'tracksdata @ git+https://github.com/royerlab/tracksdata@main'\n"
    "  pip install --no-deps <royerlab/kaggle-cell-tracking-competition checkout>"
)


def official_available() -> bool:
    """Return True iff the official scorer can be imported in this interpreter."""
    try:  # pragma: no cover - environment probe
        import tracksdata  # noqa: F401
        from tracking_cellmot.metrics import evaluate  # noqa: F401
    except Exception:
        return False
    return True


def _require_official():
    """Import and return the official ``(tracksdata, metrics)`` modules or raise."""
    try:
        import tracksdata as td
        from tracking_cellmot import metrics as official_metrics
    except Exception as exc:  # pragma: no cover - only hit when deps missing
        raise OfficialScorerUnavailable(_INSTALL_HINT) from exc
    return td, official_metrics


def _to_tracksdata(graph: TrackingGraph):
    """Build an in-memory ``tracksdata`` graph from a :class:`TrackingGraph`.

    Nodes carry ``(t, z, y, x)`` exactly as stored; edges are added in insertion
    order so the official metric's lowest-edge-id tie-breaks (merge collapse /
    out-degree cap) resolve identically to the clean-room implementation. The
    original integer node ids are preserved via the explicit ``index`` argument,
    so a caller can map results back if needed (matching is spatial, not by id,
    so this is a convenience not a correctness requirement).
    """
    import polars as pl

    td, _ = _require_official()
    K = td.DEFAULT_ATTR_KEYS

    g = td.graph.RustWorkXGraph()
    for key in (K.Z, K.Y, K.X):
        g.add_node_attr_key(key, pl.Float64, 0.0)

    id_map: dict[int, int] = {}
    for nid in graph.node_ids():
        t, z, y, x = graph.coords[nid]
        id_map[nid] = g.add_node(
            {K.T: int(round(t)), K.Z: float(z), K.Y: float(y), K.X: float(x)}
        )
    for source_id, target_id in graph.edges:
        g.add_edge(id_map[source_id], id_map[target_id], {})
    return g


def official_evaluate(
    pred: TrackingGraph,
    gt: TrackingGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> EvaluationResult:
    """Score one (pred, gt) pair with the GENUINE official metric.

    Returns the same :class:`~biohub_tracking.eval.score.EvaluationResult`
    named-tuple the clean-room :func:`biohub_tracking.eval.score.evaluate`
    returns, so the two are directly diff-able.
    """
    _, official_metrics = _require_official()
    td_pred = _to_tracksdata(pred)
    td_gt = _to_tracksdata(gt)
    er = official_metrics.evaluate(
        td_pred, td_gt, scale=scale, max_distance=max_distance
    )
    return EvaluationResult(
        edge_tp=int(er.edge_tp),
        edge_fp=int(er.edge_fp),
        edge_fn=int(er.edge_fn),
        division_tp=int(er.division_tp),
        division_fp=int(er.division_fp),
        division_fn=int(er.division_fn),
        num_pred_nodes=int(er.num_pred_nodes),
    )


def official_score_family(
    name: str,
    lineage: str,
    pred: TrackingGraph,
    gt: TrackingGraph,
    n_true: float,
    scale: tuple[float, ...] | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> FamilyResult:
    """Official-metric counterpart of :func:`biohub_tracking.eval.cv.score_family`.

    The adjusted edge Jaccard uses the official ``per_sample_metrics`` formula
    (identical algebra to the clean-room one, α = :data:`ADJUSTMENT_ALPHA`), so a
    :class:`FamilyResult` from here plugs straight into
    :func:`biohub_tracking.eval.cv.aggregate` for an apples-to-apples micro-average.
    """
    _, official_metrics = _require_official()
    r = official_evaluate(pred, gt, scale=scale, max_distance=max_distance)
    # Official per_sample_metrics derives edge_jaccard + adjusted edge Jaccard.
    row = official_metrics.per_sample_metrics(
        official_metrics.EvaluationResult(
            edge_tp=r.edge_tp,
            edge_fp=r.edge_fp,
            edge_fn=r.edge_fn,
            division_tp=r.division_tp,
            division_fp=r.division_fp,
            division_fn=r.division_fn,
            num_pred_nodes=r.num_pred_nodes,
        ),
        n_total=n_true,
        node_recall=0.0,  # unused for the score; recall is a diagnostic only
    )
    j = row["edge_jaccard"]
    adj = row["adj_edge_jaccard"]
    if isinstance(adj, float) and math.isnan(adj):
        adj = j  # no node estimate → fall back to the raw Jaccard (as clean-room)
    return FamilyResult(
        name=name,
        lineage=lineage,
        edge_tp=r.edge_tp,
        edge_fp=r.edge_fp,
        edge_fn=r.edge_fn,
        edge_jaccard=j,
        adj_edge_jaccard=adj,
        division_tp=r.division_tp,
        division_fp=r.division_fp,
        division_fn=r.division_fn,
        num_pred_nodes=r.num_pred_nodes,
        n_true=n_true,
        weight=r.edge_tp + r.edge_fp + r.edge_fn,
    )


class DivergenceRow(NamedTuple):
    """Per-family divergence between the clean-room and official scorers.

    Every ``d_*`` field is ``official − clean_room`` for that count; a faithful
    clean-room oracle drives all of them to zero.
    """

    name: str
    clean: EvaluationResult
    official: EvaluationResult
    d_edge_tp: int
    d_edge_fp: int
    d_edge_fn: int
    d_division_tp: int
    d_division_fp: int
    d_division_fn: int
    clean_edge_jaccard: float
    official_edge_jaccard: float
    d_edge_jaccard: float

    @property
    def counts_match(self) -> bool:
        return (
            self.d_edge_tp == 0
            and self.d_edge_fp == 0
            and self.d_edge_fn == 0
            and self.d_division_tp == 0
            and self.d_division_fp == 0
            and self.d_division_fn == 0
        )


def divergence_row(
    name: str,
    pred: TrackingGraph,
    gt: TrackingGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> DivergenceRow:
    """Score one pair with BOTH scorers and return the signed count divergence."""
    from .score import evaluate as clean_evaluate

    clean = clean_evaluate(pred, gt, scale=scale, max_distance=max_distance)
    official = official_evaluate(pred, gt, scale=scale, max_distance=max_distance)

    def _j(er: EvaluationResult) -> float:
        return _jaccard(er.edge_tp, er.edge_fp, er.edge_fn)

    cj, oj = _j(clean), _j(official)
    dj = (
        float("nan")
        if (math.isnan(cj) or math.isnan(oj))
        else oj - cj
    )
    return DivergenceRow(
        name=name,
        clean=clean,
        official=official,
        d_edge_tp=official.edge_tp - clean.edge_tp,
        d_edge_fp=official.edge_fp - clean.edge_fp,
        d_edge_fn=official.edge_fn - clean.edge_fn,
        d_division_tp=official.division_tp - clean.division_tp,
        d_division_fp=official.division_fp - clean.division_fp,
        d_division_fn=official.division_fn - clean.division_fn,
        clean_edge_jaccard=cj,
        official_edge_jaccard=oj,
        d_edge_jaccard=dj,
    )


def divergence_row_to_dict(row: DivergenceRow) -> dict:
    """JSON-serialisable view of a :class:`DivergenceRow` (rounded)."""

    def _er(er: EvaluationResult) -> dict:
        return {
            "edge_tp": er.edge_tp,
            "edge_fp": er.edge_fp,
            "edge_fn": er.edge_fn,
            "division_tp": er.division_tp,
            "division_fp": er.division_fp,
            "division_fn": er.division_fn,
            "num_pred_nodes": er.num_pred_nodes,
        }

    def _round(v: float) -> float | None:
        return None if (isinstance(v, float) and math.isnan(v)) else round(v, 6)

    return {
        "name": row.name,
        "counts_match": row.counts_match,
        "clean_room": _er(row.clean),
        "official": _er(row.official),
        "delta": {
            "edge_tp": row.d_edge_tp,
            "edge_fp": row.d_edge_fp,
            "edge_fn": row.d_edge_fn,
            "division_tp": row.d_division_tp,
            "division_fp": row.d_division_fp,
            "division_fn": row.d_division_fn,
        },
        "clean_edge_jaccard": _round(row.clean_edge_jaccard),
        "official_edge_jaccard": _round(row.official_edge_jaccard),
        "d_edge_jaccard": _round(row.d_edge_jaccard),
    }


# --------------------------------------------------------------------------- #
# geff <-> csv converter port (light, dependency-free — SOT-2995 scope)        #
#                                                                             #
# The official csv_to_geffs / geffs_to_csv scripts route through tracksdata's #
# save_graph / from_geff. The clean-room csv side already lives in            #
# biohub_tracking.io (load_submission_csv / write_submission_csv) and the     #
# geff *read* side in io.load_geff; the only missing leg is graph -> geff.    #
# graph_to_geff writes the exact zarr-v3 layout io.load_geff reads, so the    #
# offline pipeline can emit geffs the official scripts.evaluate consumes       #
# WITHOUT pulling tracksdata into the writer.                                  #
# --------------------------------------------------------------------------- #
def graph_to_geff(
    graph: TrackingGraph,
    geff_path,
    scale: tuple[float, float, float] | None = None,
    estimated_number_of_nodes: float | None = None,
):
    """Write a :class:`TrackingGraph` to a ``.geff`` (zarr-v3) directory.

    The light counterpart of the official ``csv_to_geffs`` step: it produces a
    geff directory that :func:`biohub_tracking.io.load_geff` round-trips exactly,
    and whose ``geff.axes`` / ``geff.extra.estimated_number_of_nodes`` metadata
    matches what :func:`biohub_tracking.io.geff_scale` /
    :func:`biohub_tracking.io.geff_estimated_num_nodes` read back.
    """
    from pathlib import Path

    import numpy as np
    import zarr

    geff_path = Path(geff_path)
    node_ids = graph.node_ids()
    ts = np.array([int(round(graph.coords[n][0])) for n in node_ids], dtype="int64")
    zs = np.array([graph.coords[n][1] for n in node_ids], dtype="int64")
    ys = np.array([graph.coords[n][2] for n in node_ids], dtype="int64")
    xs = np.array([graph.coords[n][3] for n in node_ids], dtype="int64")
    ids = np.array([int(n) for n in node_ids], dtype="uint64")
    edges = (
        np.array(graph.edges, dtype="uint64")
        if graph.edges
        else np.zeros((0, 2), dtype="uint64")
    )

    root = zarr.open_group(str(geff_path), mode="w")
    axes = [{"name": "t", "type": "time"}]
    sc = scale if scale is not None else (1.0, 1.0, 1.0)
    for axis_name, axis_scale in zip(("z", "y", "x"), sc):
        axes.append({"name": axis_name, "type": "space", "scale": float(axis_scale)})
    geff_meta: dict = {"axes": axes}
    if estimated_number_of_nodes is not None and not (
        isinstance(estimated_number_of_nodes, float)
        and math.isnan(estimated_number_of_nodes)
    ):
        geff_meta["extra"] = {
            "estimated_number_of_nodes": float(estimated_number_of_nodes)
        }
    root.attrs["geff"] = geff_meta

    root.create_array("nodes/ids", shape=ids.shape, dtype="uint64")[:] = ids
    for prop, arr in (("t", ts), ("z", zs), ("y", ys), ("x", xs)):
        root.create_array(
            f"nodes/props/{prop}/values", shape=arr.shape, dtype="int64"
        )[:] = arr
    root.create_array("edges/ids", shape=edges.shape, dtype="uint64")[:] = edges
    return geff_path
