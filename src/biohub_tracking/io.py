"""I/O for the two tracking-graph on-disk formats.

Ground truth ships as **geff** (Graph Exchange File Format, a zarr-v3 directory);
predictions are submitted as a single **flat CSV** where each row is either a
node or an edge. This module converts both to/from :class:`TrackingGraph`.

Submission CSV columns (one header row, ``id`` is a 0-based row index)::

    id, dataset, row_type, node_id, t, z, y, x, source_id, target_id

* ``row_type == "node"``: ``node_id, t, z, y, x`` are set; ``source_id`` and
  ``target_id`` are ``-1``.
* ``row_type == "edge"``: ``source_id, target_id`` are set; the node columns are
  ``-1``.

geff layout (per dataset ``<name>.geff/``)::

    zarr.json                       # metadata: geff.axes (scales), extra
    nodes/ids                       # uint64 node ids, shape (N,)
    nodes/props/{t,z,y,x}/values    # int64 per-node property arrays, shape (N,)
    edges/ids                       # uint64 (source, target) pairs, shape (E, 2)
"""

from __future__ import annotations

import csv
from pathlib import Path

from .graph import TrackingGraph
from .matching import DEFAULT_SCALE

CSV_COLUMNS: tuple[str, ...] = (
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
)


# --------------------------------------------------------------------------- #
# geff (ground truth)                                                          #
# --------------------------------------------------------------------------- #
def geff_scale(geff_path: Path | str) -> tuple[float, float, float]:
    """Return the ``(z, y, x)`` physical voxel scale recorded in a geff file."""
    import zarr

    root = zarr.open(str(geff_path), mode="r")
    axes = root.attrs.get("geff", {}).get("axes", [])
    by_name = {a["name"]: a.get("scale", 1.0) for a in axes}
    if all(k in by_name for k in ("z", "y", "x")):
        return (float(by_name["z"]), float(by_name["y"]), float(by_name["x"]))
    return DEFAULT_SCALE


def geff_estimated_num_nodes(geff_path: Path | str) -> float:
    """Return the coarse true-node-count estimate from geff metadata, or NaN."""
    import zarr

    root = zarr.open(str(geff_path), mode="r")
    extra = root.attrs.get("geff", {}).get("extra", {}) or {}
    val = extra.get("estimated_number_of_nodes")
    return float(val) if val is not None else float("nan")


def load_geff(geff_path: Path | str) -> TrackingGraph:
    """Load a geff directory into a :class:`TrackingGraph`."""
    import zarr

    root = zarr.open(str(geff_path), mode="r")
    node_ids = root["nodes/ids"][:]
    t = root["nodes/props/t/values"][:]
    z = root["nodes/props/z/values"][:]
    y = root["nodes/props/y/values"][:]
    x = root["nodes/props/x/values"][:]

    graph = TrackingGraph()
    for nid, tt, zz, yy, xx in zip(node_ids, t, z, y, x):
        graph.add_node(int(nid), float(tt), float(zz), float(yy), float(xx))

    edge_ids = root["edges/ids"][:]
    for source_id, target_id in edge_ids:
        graph.add_edge(int(source_id), int(target_id))

    return graph


def load_geff_dir(data_dir: Path | str) -> dict[str, TrackingGraph]:
    """Load every ``*.geff`` in *data_dir* keyed by dataset name (the file stem)."""
    data_dir = Path(data_dir)
    return {p.stem: load_geff(p) for p in sorted(data_dir.glob("*.geff"))}


# --------------------------------------------------------------------------- #
# submission CSV (predictions)                                                #
# --------------------------------------------------------------------------- #
def load_submission_csv(csv_path: Path | str) -> dict[str, TrackingGraph]:
    """Parse a submission CSV into ``{dataset: TrackingGraph}``.

    Node rows are read before edges within a dataset are attached, so a file that
    lists all nodes before all edges (the canonical layout) loads correctly
    regardless of interleaving.
    """
    node_rows: dict[str, list[tuple[int, float, float, float, float]]] = {}
    edge_rows: dict[str, list[tuple[int, int]]] = {}

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            dataset = row["dataset"]
            if row["row_type"] == "node":
                node_rows.setdefault(dataset, []).append(
                    (
                        int(row["node_id"]),
                        float(row["t"]),
                        float(row["z"]),
                        float(row["y"]),
                        float(row["x"]),
                    )
                )
            elif row["row_type"] == "edge":
                edge_rows.setdefault(dataset, []).append(
                    (int(row["source_id"]), int(row["target_id"]))
                )
            else:
                raise ValueError(f"unknown row_type {row['row_type']!r}")

    graphs: dict[str, TrackingGraph] = {}
    for dataset in set(node_rows) | set(edge_rows):
        g = TrackingGraph()
        for nid, t, z, y, x in node_rows.get(dataset, []):
            g.add_node(nid, t, z, y, x)
        for source_id, target_id in edge_rows.get(dataset, []):
            g.add_edge(source_id, target_id)
        graphs[dataset] = g
    return graphs


def graph_to_rows(graph: TrackingGraph, dataset: str) -> list[dict]:
    """Flatten one graph into submission rows: all nodes, then all edges."""
    rows: list[dict] = []
    for nid in graph.node_ids():
        t, z, y, x = graph.coords[nid]
        rows.append(
            {
                "dataset": dataset,
                "row_type": "node",
                "node_id": nid,
                "t": int(round(t)),
                "z": int(round(z)),
                "y": int(round(y)),
                "x": int(round(x)),
                "source_id": -1,
                "target_id": -1,
            }
        )
    for source_id, target_id in graph.edges:
        rows.append(
            {
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": source_id,
                "target_id": target_id,
            }
        )
    return rows


def write_submission_csv(
    graphs: dict[str, TrackingGraph], csv_path: Path | str
) -> Path:
    """Write ``{dataset: graph}`` to a submission CSV; return the path."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        idx = 0
        for dataset in sorted(graphs):
            for row in graph_to_rows(graphs[dataset], dataset):
                writer.writerow(
                    [
                        idx,
                        row["dataset"],
                        row["row_type"],
                        row["node_id"],
                        row["t"],
                        row["z"],
                        row["y"],
                        row["x"],
                        row["source_id"],
                        row["target_id"],
                    ]
                )
                idx += 1
    return csv_path
