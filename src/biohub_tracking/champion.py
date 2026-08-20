"""Load the reigning champion detection+linking configuration.

The champion is stored declaratively in ``registry.json`` / ``champion/config.json``
at the repo root so the promotion state is data, not code. This module reads that
config and constructs the :class:`~biohub_tracking.detect.DetectParams` /
:class:`~biohub_tracking.link.LinkParams` the pipeline runs with, giving later
stages (e.g. submission building) a single source of truth for "what is champion".

**Exec / kernel compatibility (SOT-1984).** The champion params must be
resolvable even when the code runs in a stripped environment — inside a Kaggle
submission kernel, under ``exec()`` with no ``__file__`` bound, or from an
arbitrary working directory. So config resolution never assumes ``__file__`` is
defined and falls back, in order, to: an explicit path → the
``BIOHUB_CHAMPION_CONFIG`` env var → the file next to this module → the current
working directory → an **embedded copy** of the frozen champion (below). The
embedded copy is kept byte-for-byte in sync with ``champion/config.json`` by
``tests/test_exec_compat.py`` so the pipeline still runs with zero filesystem.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .detect import DEFAULT_SCALE, DetectParams
from .link import LinkParams

# Frozen champion parameters, embedded so the pipeline runs with no filesystem
# access (Kaggle kernel / exec()). MUST mirror champion/config.json exactly;
# tests/test_exec_compat.py asserts they stay identical.
EMBEDDED_CHAMPION_CONFIG: dict = {
    "name": "detect-link-dog-v4-shorttrack",
    "description": (
        "DoG-v3 adaptive detection (median + 3.0*1.4826*MAD of the local-contrast "
        "response) + optimal nearest-neighbour frame linking, now with post-link "
        "short-track pruning (min_track_length=4). Fourth biohub-claude champion "
        "(SOT-2369): a detection that never links into a >=4-node track is almost "
        "always noise, so pruning those weakly-connected fragments both relieves the "
        "node-count penalty AND frees GT nodes so the per-timepoint <=7um matching "
        "attaches the persistent track instead of a transient decoy. On the SOT-2305 "
        "4-dataset LB holdout every dataset improves with no regression (44b6_0113de3b "
        "0.8814->0.8895, 44b6_0b24845f 0.6658->0.6817, 6bba_05b6850b 0.5025->0.5700 "
        "with edge TP 619->651/FP 251->215/FN 226->194, 6bba_05db0fb1 0.7096->0.7310), "
        "raising the holdout micro-adj from 0.6232 to 0.6649 (+0.042). "
        "min_track_length=5 scored a hair higher on micro (0.6692) but regressed the "
        "clean 44b6_0113de3b family (TP 47->45, adj 0.8895->0.8539) by pruning real "
        "short tracks, so mtl=4 is chosen for no-per-dataset-regression robustness. "
        "Ported from the public frontier lineage tracker's FILTER_SHORT_TRACKS "
        "post-processing (that notebook's 0.913 comes from a GPU pretrained UNet+ILP "
        "pipeline that cannot run under this repo's numpy/scipy/zarr, CPU, no-internet, "
        "no-weights kernel; the short-track filter is the one score lever that transfers)."
    ),
    "scale": [1.625, 0.40625, 0.40625],
    "detect": {
        "sigma_zyx": [1.0, 2.0, 2.0],
        "background_sigma_zyx": [2.0, 6.0, 6.0],
        "nms_size_zyx": [2, 5, 5],
        "threshold_percentile": 92.0,
        "mad_k": 3.0,
        "min_threshold": 0.0,
    },
    "link": {
        "max_distance": 7.0,
        "allow_division": False,
        "division_distance": 7.0,
        "min_track_length": 4,
    },
}

_CONFIG_ENV_VAR = "BIOHUB_CHAMPION_CONFIG"


def _module_relative_config() -> Path | None:
    """``champion/config.json`` relative to this file, or ``None`` if ``__file__``
    is not bound (running under ``exec()`` / in a kernel cell)."""
    try:
        here = Path(__file__).resolve()
    except NameError:  # pragma: no cover - only when __file__ is unbound
        return None
    return here.parents[2] / "champion" / "config.json"


def _candidate_config_paths() -> list[Path]:
    """Ordered on-disk locations to probe for the champion config."""
    candidates: list[Path] = []
    env = os.environ.get(_CONFIG_ENV_VAR)
    if env:
        candidates.append(Path(env))
    module_rel = _module_relative_config()
    if module_rel is not None:
        candidates.append(module_rel)
    # cwd-relative fallbacks: run from the repo root or from champion/.
    cwd = Path.cwd()
    candidates.append(cwd / "champion" / "config.json")
    candidates.append(cwd / "config.json")
    return candidates


# Best-effort static path (may be None under exec()); kept for backwards compat.
CHAMPION_CONFIG_PATH = _module_relative_config()


def load_champion_config(path: Path | str | None = None) -> dict:
    """Return the champion config dict.

    Resolution order (exec/kernel-safe): explicit *path* → ``BIOHUB_CHAMPION_CONFIG``
    env var → the file next to this module → the current working directory → the
    :data:`EMBEDDED_CHAMPION_CONFIG` fallback. The embedded fallback guarantees a
    usable champion even with no filesystem (a Kaggle kernel), so this never
    raises for a missing file.
    """
    if path is not None:
        with open(path) as fh:
            return json.load(fh)
    for candidate in _candidate_config_paths():
        try:
            with open(candidate) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            continue
    # No file found anywhere — fall back to the embedded frozen champion.
    return json.loads(json.dumps(EMBEDDED_CHAMPION_CONFIG))


def champion_params(
    config: dict | None = None,
) -> tuple[DetectParams, LinkParams, tuple[float, float, float]]:
    """Build ``(DetectParams, LinkParams, scale)`` from a champion config."""
    if config is None:
        config = load_champion_config()
    d = config["detect"]
    l = config["link"]
    bg = d.get("background_sigma_zyx")
    mad_k = d.get("mad_k")
    inorm = d.get("intensity_norm")
    detect = DetectParams(
        sigma_zyx=tuple(d.get("sigma_zyx", (1.0, 3.0, 3.0))),
        nms_size_zyx=tuple(d.get("nms_size_zyx", (2, 5, 5))),
        threshold_percentile=float(d.get("threshold_percentile", 99.3)),
        min_threshold=float(d.get("min_threshold", 0.0)),
        background_sigma_zyx=tuple(bg) if bg is not None else None,
        mad_k=None if mad_k is None else float(mad_k),
        intensity_norm=(
            (str(inorm[0]), float(inorm[1]), float(inorm[2]))
            if inorm is not None
            else None
        ),
    )
    link = LinkParams(
        max_distance=float(l.get("max_distance", 7.0)),
        allow_division=bool(l.get("allow_division", False)),
        division_distance=float(l.get("division_distance", 7.0)),
        division_max_sibling_ratio=float(l.get("division_max_sibling_ratio", 0.0)),
        velocity_gain=float(l.get("velocity_gain", 0.0)),
        velocity_disp_weight=float(l.get("velocity_disp_weight", 0.05)),
        motion_gate_on_prediction=bool(l.get("motion_gate_on_prediction", False)),
        max_frame_gap=int(l.get("max_frame_gap", 1)),
        gap_distance=float(l.get("gap_distance", 7.0)),
        min_track_length=int(l.get("min_track_length", 1)),
    )
    scale = tuple(config.get("scale", DEFAULT_SCALE))
    return detect, link, scale
