"""Load the reigning champion detection+linking configuration.

The champion is stored declaratively in ``registry.json`` / ``champion/config.json``
at the repo root so the promotion state is data, not code. This module reads that
config and constructs the :class:`~biohub_tracking.detect.DetectParams` /
:class:`~biohub_tracking.link.LinkParams` the pipeline runs with, giving later
stages (e.g. submission building) a single source of truth for "what is champion".
"""

from __future__ import annotations

import json
from pathlib import Path

from .detect import DEFAULT_SCALE, DetectParams
from .link import LinkParams

# Repo root = two levels up from this file (src/biohub_tracking/champion.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
CHAMPION_CONFIG_PATH = _REPO_ROOT / "champion" / "config.json"


def load_champion_config(path: Path | str | None = None) -> dict:
    """Return the champion config dict from ``champion/config.json``."""
    path = Path(path) if path is not None else CHAMPION_CONFIG_PATH
    with open(path) as fh:
        return json.load(fh)


def champion_params(
    config: dict | None = None,
) -> tuple[DetectParams, LinkParams, tuple[float, float, float]]:
    """Build ``(DetectParams, LinkParams, scale)`` from a champion config."""
    if config is None:
        config = load_champion_config()
    d = config["detect"]
    l = config["link"]
    detect = DetectParams(
        sigma_zyx=tuple(d.get("sigma_zyx", (1.0, 3.0, 3.0))),
        nms_size_zyx=tuple(d.get("nms_size_zyx", (2, 5, 5))),
        threshold_percentile=float(d.get("threshold_percentile", 99.3)),
        min_threshold=float(d.get("min_threshold", 0.0)),
    )
    link = LinkParams(
        max_distance=float(l.get("max_distance", 7.0)),
        allow_division=bool(l.get("allow_division", False)),
        division_distance=float(l.get("division_distance", 7.0)),
    )
    scale = tuple(config.get("scale", DEFAULT_SCALE))
    return detect, link, scale
