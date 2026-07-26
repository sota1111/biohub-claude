"""End-to-end detection + linking pipeline for one OME-NGFF video.

Reads an OME-NGFF ``*.zarr`` volume ``(T, Z, Y, X)``, detects centroids per
timepoint (:mod:`biohub_tracking.detect`) and links them into a tracking graph
(:mod:`biohub_tracking.link`). The result is a
:class:`~biohub_tracking.graph.TrackingGraph` ready to be scored locally or
written to a submission CSV.
"""

from __future__ import annotations

from pathlib import Path

from .detect import DEFAULT_SCALE, DetectParams, detect_volume_series
from .graph import TrackingGraph
from .link import LinkParams, link_centroids


def _open_image_array(zarr_path: Path | str):
    """Open the level-0 array of an OME-NGFF ``*.zarr`` group (falls back to array)."""
    import zarr

    root = zarr.open(str(zarr_path), mode="r")
    if hasattr(root, "shape") and not hasattr(root, "keys"):
        return root  # already an array
    # OME multiscales: level "0" is full resolution.
    return root["0"]


def run_pipeline(
    zarr_path: Path | str,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    detect_params: DetectParams | None = None,
    link_params: LinkParams | None = None,
    max_t: int | None = None,
) -> TrackingGraph:
    """Detect and link one video into a tracking graph."""
    arr = _open_image_array(zarr_path)
    detections = detect_volume_series(arr, detect_params, max_t=max_t)
    return link_centroids(detections, scale=scale, params=link_params)
