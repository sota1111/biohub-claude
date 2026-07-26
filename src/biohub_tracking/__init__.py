"""biohub_tracking — data loader + local evaluation for the Biohub cell-tracking
Kaggle competition (``biohub-cell-tracking-during-development``).

Public surface:

* :class:`~biohub_tracking.graph.TrackingGraph` — in-memory tracking graph.
* :mod:`biohub_tracking.io` — geff (ground truth) and submission-CSV I/O.
* :mod:`biohub_tracking.matching` — per-timepoint ≤7 µm bipartite node matching.
* :mod:`biohub_tracking.eval` — Edge/Division Jaccard and the combined score.
"""

from .graph import TrackingGraph

__all__ = ["TrackingGraph"]
__version__ = "0.1.0"
