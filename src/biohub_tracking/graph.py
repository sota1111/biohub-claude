"""In-memory tracking-graph representation.

A tracking graph has **nodes** (cell detections, each with a timepoint ``t`` and
a 3D centroid ``(z, y, x)``) and directed **edges** linking a cell at one
timepoint to the same cell — or, at a division, its two daughters — at the next
timepoint. A biological *division* is a node with exactly two outgoing edges.

This module is deliberately dependency-light (pure Python / numpy): the whole
local evaluation stack is built on it so it can run unchanged inside a Kaggle
submission kernel.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

# Order of the spatial coordinate columns everywhere in this package.
SPATIAL_AXES: tuple[str, str, str] = ("z", "y", "x")


@dataclass
class TrackingGraph:
    """A directed tracking graph keyed by integer node id.

    Attributes
    ----------
    coords
        ``node_id -> (t, z, y, x)`` as a float tuple. ``t`` is the integer
        timepoint stored as a float for convenience.
    _succ / _pred
        Adjacency lists preserving edge-insertion order (edge ids are the index
        of the edge in :attr:`edges`, so a stable order matters for the
        tie-breaks the metric applies, e.g. capping out-degree at two).
    edges
        Directed ``(source_id, target_id)`` pairs in insertion order.
    """

    coords: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    _succ: dict[int, list[int]] = field(default_factory=dict)
    _pred: dict[int, list[int]] = field(default_factory=dict)
    edges: list[tuple[int, int]] = field(default_factory=list)

    # -- construction -----------------------------------------------------
    def add_node(self, node_id: int, t: float, z: float, y: float, x: float) -> None:
        node_id = int(node_id)
        if node_id in self.coords:
            raise ValueError(f"duplicate node id {node_id}")
        self.coords[node_id] = (float(t), float(z), float(y), float(x))
        self._succ.setdefault(node_id, [])
        self._pred.setdefault(node_id, [])

    def add_edge(self, source_id: int, target_id: int) -> None:
        source_id, target_id = int(source_id), int(target_id)
        if source_id not in self.coords or target_id not in self.coords:
            raise KeyError(f"edge {(source_id, target_id)} references an unknown node")
        self.edges.append((source_id, target_id))
        self._succ[source_id].append(target_id)
        self._pred[target_id].append(source_id)

    # -- queries ----------------------------------------------------------
    def node_ids(self) -> list[int]:
        return list(self.coords)

    @property
    def num_nodes(self) -> int:
        return len(self.coords)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def t(self, node_id: int) -> int:
        return int(self.coords[node_id][0])

    def position(self, node_id: int) -> np.ndarray:
        """Return the ``(z, y, x)`` centroid as a float array (voxel units)."""
        return np.asarray(self.coords[node_id][1:], dtype=float)

    def successors(self, node_id: int) -> list[int]:
        return list(self._succ.get(node_id, ()))

    def predecessors(self, node_id: int) -> list[int]:
        return list(self._pred.get(node_id, ()))

    def out_degree(self, node_id: int) -> int:
        return len(self._succ.get(node_id, ()))

    def in_degree(self, node_id: int) -> int:
        return len(self._pred.get(node_id, ()))

    def dividing_nodes(self) -> list[int]:
        """Nodes with at least two outgoing edges (forks / divisions)."""
        return [n for n in self.coords if self.out_degree(n) >= 2]

    def nodes_by_time(self) -> dict[int, list[int]]:
        """Group node ids by integer timepoint."""
        out: dict[int, list[int]] = {}
        for node_id, (t, *_rest) in self.coords.items():
            out.setdefault(int(t), []).append(node_id)
        return out

    def subgraph(self, node_ids: "Iterable[int]") -> "TrackingGraph":
        """Return the subgraph induced on *node_ids* (edges with both ends kept)."""
        keep = set(int(n) for n in node_ids)
        sub = TrackingGraph()
        for node_id in self.coords:
            if node_id in keep:
                t, z, y, x = self.coords[node_id]
                sub.add_node(node_id, t, z, y, x)
        for source_id, target_id in self.edges:
            if source_id in keep and target_id in keep:
                sub.add_edge(source_id, target_id)
        return sub

    @classmethod
    def from_lists(
        cls,
        nodes: dict[int, tuple[float, float, float, float]],
        edges: list[tuple[int, int]],
    ) -> "TrackingGraph":
        """Build a graph from ``{node_id: (t, z, y, x)}`` and an edge list."""
        g = cls()
        for node_id, (t, z, y, x) in nodes.items():
            g.add_node(node_id, t, z, y, x)
        for source_id, target_id in edges:
            g.add_edge(source_id, target_id)
        return g
