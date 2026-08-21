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


def learned_detector_config(config: dict | None = None) -> dict | None:
    """Return the ``learned_detector`` receptacle block (SOT-2847), or ``None``.

    The learned detector is a **whole-stage** replacement of classical detection,
    not a :class:`DetectParams` knob, so it is resolved separately from
    :func:`champion_params` and read by the submission/pipeline layer. The
    champion config carries no such block, so this returns ``None`` and the
    classical champion path is byte-for-byte unchanged. Accepts the block either
    at the config top level or nested under ``detect`` (either placement is
    honoured); an absent or non-dict value yields ``None``.
    """
    if config is None:
        config = load_champion_config()
    block = config.get("learned_detector")
    if block is None:
        detect = config.get("detect")
        if isinstance(detect, dict):
            block = detect.get("learned_detector")
    return block if isinstance(block, dict) else None


def champion_params(
    config: dict | None = None,
) -> tuple[DetectParams, LinkParams, tuple[float, float, float]]:
    """Build ``(DetectParams, LinkParams, scale)`` from a champion config."""
    if config is None:
        config = load_champion_config()
    d = config["detect"]
    l = config["link"]
    overlay = l.get("division_overlay")
    bg = d.get("background_sigma_zyx")
    mad_k = d.get("mad_k")
    inorm = d.get("intensity_norm")
    wshed = d.get("watershed")
    dscales = d.get("dog_scales_zyx")
    bfilter = d.get("blobness_filter")
    dsplit = d.get("density_gated_split")
    lthr = d.get("local_threshold")
    dscorer = d.get("detect_scorer")
    rrec = d.get("recall_recovery")
    hsel = d.get("detect_hypothesis_select")
    hsel_low = d.get("hypothesis_mad_k_low")
    hsel_min_track = d.get("hypothesis_min_track")
    hsel_pool_cap = d.get("hypothesis_pool_cap")
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
        watershed=(
            (str(wshed[0]), float(wshed[1]), float(wshed[2]), float(wshed[3]))
            if wshed is not None
            else None
        ),
        dog_scales_zyx=(
            tuple(tuple(float(v) for v in s) for s in dscales)
            if dscales is not None
            else None
        ),
        blobness_filter=(
            (
                str(bfilter[0]),
                tuple(float(v) for v in bfilter[1]),
                float(bfilter[2]),
                float(bfilter[3]),
            )
            if bfilter is not None
            else None
        ),
        density_gated_split=(
            (
                str(dsplit[0]),
                float(dsplit[1]),
                float(dsplit[2]),
                float(dsplit[3]),
                float(dsplit[4]),
                float(dsplit[5]),
            )
            if dsplit is not None
            else None
        ),
        local_threshold=(
            (str(lthr[0]), tuple(int(v) for v in lthr[1]), float(lthr[2]))
            if lthr is not None
            else None
        ),
        # Learned detection scorer (SOT-2828): the embedded-coefficient dict passes
        # through as-is; DetectParams.detect_scorer=None (absent) keeps the champion
        # byte-identical.
        detect_scorer=dict(dscorer) if dscorer is not None else None,
        # Recall-oriented FN-edge-endpoint recovery tier (SOT-2873); absent key
        # keeps the champion byte-identical (recall_recovery default off).
        recall_recovery=(
            (str(rrec[0]), float(rrec[1]), float(rrec[2]))
            if rrec is not None
            else None
        ),
        # Ultrack multi-hypothesis detection selection by temporal support
        # (SOT-2884); absent keys keep the champion byte-identical (flag off).
        detect_hypothesis_select=bool(hsel) if hsel is not None else False,
        hypothesis_mad_k_low=None if hsel_low is None else float(hsel_low),
        hypothesis_min_track=(3 if hsel_min_track is None else int(hsel_min_track)),
        hypothesis_pool_cap=(
            20000 if hsel_pool_cap is None else int(hsel_pool_cap)
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
        # ARGUS motion-model predicted-position LAP linking (SOT-2864); absent keys
        # keep the champion byte-identical (motion_model_link default off).
        motion_model_link=bool(l.get("motion_model_link", False)),
        motion_smooth_sigma=float(l.get("motion_smooth_sigma", 15.0)),
        motion_gain=float(l.get("motion_gain", 1.0)),
        # Ultrack bidirectional motion-consistency gate (SOT-2883); absent keys keep
        # the champion byte-identical (link_consistency_gate default off). JSON cannot
        # hold inf, so an absent/null tol means "no hard rejection" (soft penalty only).
        link_consistency_gate=bool(l.get("link_consistency_gate", False)),
        link_consistency_tol=(
            float("inf")
            if l.get("link_consistency_tol") is None
            else float(l["link_consistency_tol"])
        ),
        link_consistency_weight=float(l.get("link_consistency_weight", 0.0)),
        max_frame_gap=int(l.get("max_frame_gap", 1)),
        gap_distance=float(l.get("gap_distance", 7.0)),
        # Node-interpolation gap recovery (SOT-2849); absent keys keep the champion
        # byte-identical (gap_recover default off).
        gap_recover=bool(l.get("gap_recover", False)),
        gap_recover_max_gap=int(l.get("gap_recover_max_gap", 2)),
        gap_recover_distance=float(l.get("gap_recover_distance", 7.0)),
        gap_recover_min_frag=int(l.get("gap_recover_min_frag", 1)),
        global_window=int(l.get("global_window", 1)),
        # birth/death arc costs (SOT-2830); JSON cannot hold inf, so an absent or
        # null value means "unbounded" (never refuse a feasible link => the global
        # path reproduces the per-frame champion matching).
        birth_cost=(
            float("inf") if l.get("birth_cost") is None else float(l["birth_cost"])
        ),
        death_cost=(
            float("inf") if l.get("death_cost") is None else float(l["death_cost"])
        ),
        appearance_weight=float(l.get("appearance_weight", 0.0)),
        # Learned edge-linking cost (SOT-2841): the embedded-coefficient dict passes
        # through as-is; LinkParams.edge_cost_model=None (absent) keeps the champion
        # byte-identical.
        edge_cost_model=(
            dict(l["edge_cost_model"]) if l.get("edge_cost_model") is not None else None
        ),
        # Motion-coupled windowed global association (SOT-2871); absent keys keep the
        # champion byte-identical (window_assoc default 1 => per-frame champion path).
        window_assoc=int(l.get("window_assoc", 1)),
        window_theta=(
            float("inf") if l.get("window_theta") is None else float(l["window_theta"])
        ),
        window_carry_weight=float(l.get("window_carry_weight", 0.5)),
        window_parental_softmax=bool(l.get("window_parental_softmax", False)),
        window_softmax_min_share=float(l.get("window_softmax_min_share", 0.3)),
        window_softmax_temp=float(l.get("window_softmax_temp", 1.0)),
        # Post-hoc suspicious-tracking-event review gate (SOT-2895); absent keys keep
        # the champion byte-identical (suspicious_review default off).
        suspicious_review=bool(l.get("suspicious_review", False)),
        suspicious_turn_cos=float(l.get("suspicious_turn_cos", -0.5)),
        suspicious_jump_ratio=float(l.get("suspicious_jump_ratio", 3.0)),
        suspicious_jump_floor=float(l.get("suspicious_jump_floor", 1.0)),
        # Two-pass tight-then-full-gate Hungarian linking (SOT-2899, classical
        # baseline port); absent keys keep the champion byte-identical (two-pass off,
        # and link_full_distance defaults to max_distance => Pass 2 a no-op).
        link_two_pass=bool(l.get("link_two_pass", False)),
        link_full_distance=float(
            l.get("link_full_distance", l.get("max_distance", 7.0))
        ),
        min_track_length=int(l.get("min_track_length", 1)),
        division_overlay=(
            (
                str(overlay[0]),
                float(overlay[1]),
                float(overlay[2]),
                int(overlay[3]),
                bool(overlay[4]),
            )
            if overlay is not None
            else None
        ),
    )
    scale = tuple(config.get("scale", DEFAULT_SCALE))
    return detect, link, scale
