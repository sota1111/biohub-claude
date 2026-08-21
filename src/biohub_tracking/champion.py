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
    "name": "detect-link-dog-v4-shorttrack-motion-gain1",
    "description": (
        "Fifth biohub-claude champion (SOT-2909 cycle-3 parent-resume promotion). "
        "Identical to the prior detect-link-dog-v4-shorttrack champion EXCEPT the "
        "linking stage now runs SOT-2864's ARGUS-style motion-model predicted-position "
        "LAP linking (motion_model_link=true, motion_smooth_sigma=15.0 um, "
        "motion_gain=1.0, motion_gate_on_prediction=true); detection knobs are "
        "byte-identical (linking-only change). PROMOTED because the two-signal gate "
        "FIRED: this exact linking config was pushed as reserve LB probe 55662947 "
        "(kernel sota1111/biohub-claude-candidate-sot-2864-motion-link v2, artifact "
        "sha256 95b8893a) and scored public 0.626 on the hidden leaderboard, i.e. it "
        "BEATS the reigning champion's genuine public best 0.624 (SOT-2369 sha256 "
        "48b1eaa2) and far exceeds the last champion submission 0.509 (SOT-2902 "
        "fingerprint-flip artifact). Simultaneously it is CV-superior on the "
        "SOT-2817/SOT-2903 re-anchored leak-free CV (primary micro_adj edge Jaccard "
        "0.6649 -> 0.6760, +0.0111, dTP+16/dFP-20/dFN-16) with 4/4 per-dataset "
        "NON-REGRESSION and micro/macro/lineage-macro all rising together "
        "(44b6_0113de3b 0.8895->0.9078, 44b6_0b24845f 0.6817->0.6938, 6bba_05b6850b "
        "0.5700->0.5748, 6bba_05db0fb1 0.7310->0.7477; macro 0.7180->0.731, "
        "lineage-macro 0.7216->0.7351), so it is NOT family-mix-sensitive. This "
        "resolves the documented CV<->LB divergence guard (SOT-2816 CV +0.042 -> "
        "public -0.115) that kept the champion byte-frozen: the CV gain of this lever "
        "DOES transfer to public, so the byte-freeze is lifted for THIS lever only. "
        "The stronger-CV motion_gain=2.0 sibling "
        "(champion/candidates/sot2900-motion-model-link-gain2.json, CV 0.6821) has "
        "UNOBSERVED public and is HELD as the #1 reserve for a single converge-phase "
        "public check (explore phase consumes public sparsely). Pure numpy/scipy, "
        "CPU, offline, no-weights, deterministic."
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
        "motion_model_link": True,
        "motion_smooth_sigma": 15.0,
        "motion_gain": 1.0,
        "motion_gate_on_prediction": True,
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


def regime_conditional_policy(config: dict | None = None):
    """Return the ``regime_conditional_detect`` policy (SOT-2923), or ``None``.

    The regime-conditional detection operating point selects ``mad_k`` /
    ``min_track_length`` **per sequence** from the observable density covariate
    (SOT-2921), so it is not a static :class:`DetectParams` / :class:`LinkParams`
    knob — it is resolved separately, like :func:`learned_detector_config`, and
    read by the submission/pipeline layer that has the per-sequence detection
    point cloud in hand. The champion config carries no such block, so this
    returns ``None`` and the classical champion path is byte-for-byte unchanged
    (default-off). Accepts the block at the config top level or nested under
    ``detect``; an absent or non-dict value yields ``None``.

    Returns a :class:`biohub_tracking.eval.regime_op.ConditionalPolicy` when the
    block is present (imported lazily so the frozen champion path never pulls in
    the eval stack).
    """
    if config is None:
        config = load_champion_config()
    block = config.get("regime_conditional_detect")
    if block is None:
        detect = config.get("detect")
        if isinstance(detect, dict):
            block = detect.get("regime_conditional_detect")
    if not isinstance(block, dict):
        return None
    from .eval.regime_op import ConditionalPolicy

    return ConditionalPolicy.from_dict(block)


def regime_conditional_link_policy(config: dict | None = None):
    """Return the ``regime_conditional_link`` policy (SOT-2922), or ``None``.

    The regime-conditional *linking* operating point selects the linking levers
    (``motion_gain`` + mutual-NN ``cycle_consistency_gate``/``margin``) **per
    sequence** from the observable density covariate (SOT-2921), so — like
    :func:`regime_conditional_policy` (detection side, SOT-2923) — it is not a
    static :class:`LinkParams` knob. It is resolved separately and read by the
    submission/pipeline layer that has the per-sequence detection point cloud in
    hand. The champion config carries no such block, so this returns ``None`` and
    the classical champion linking path is byte-for-byte unchanged (default-off).
    Accepts the block at the config top level or nested under ``link``; an absent
    or non-dict value yields ``None``.

    Returns a :class:`biohub_tracking.eval.regime_link.ConditionalLinkPolicy` when
    the block is present (imported lazily so the frozen champion path never pulls
    in the eval stack).
    """
    if config is None:
        config = load_champion_config()
    block = config.get("regime_conditional_link")
    if block is None:
        link = config.get("link")
        if isinstance(link, dict):
            block = link.get("regime_conditional_link")
    if not isinstance(block, dict):
        return None
    from .eval.regime_link import ConditionalLinkPolicy

    return ConditionalLinkPolicy.from_dict(block)


def operating_point_mixture_policy(config: dict | None = None):
    """Return the ``operating_point_mixture`` policy (SOT-2931), or ``None``.

    The soft per-sequence operating-point *mixture* selects a **continuously
    blended** linking operating point (a ``w``-weighted average of a conservative
    and an aggressive endpoint, ``w`` a logistic function of the observable
    density covariate) — the soft-mixture reformulation of the **hard** regime
    label of :func:`regime_conditional_link_policy` (SOT-2922). Like the other
    regime layers it is not a static :class:`LinkParams` knob: it is resolved
    separately and read by the layer that has the per-sequence detection point
    cloud in hand. The champion config carries no such block, so this returns
    ``None`` and the champion linking path is byte-for-byte unchanged
    (default-off). Accepts the block at the config top level or nested under
    ``link``; an absent or non-dict value yields ``None``.

    Returns a :class:`biohub_tracking.eval.regime_blend.SoftBlendPolicy` when the
    block is present (imported lazily so the frozen champion path never pulls in
    the eval stack).
    """
    if config is None:
        config = load_champion_config()
    block = config.get("operating_point_mixture")
    if block is None:
        link = config.get("link")
        if isinstance(link, dict):
            block = link.get("operating_point_mixture")
    if not isinstance(block, dict):
        return None
    from .eval.regime_blend import SoftBlendPolicy

    return SoftBlendPolicy.from_dict(block)


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
        # Local neighbour-affine motion prediction (SOT-2911); absent keys keep the
        # champion byte-identical (local_affine_predict default off).
        local_affine_predict=bool(l.get("local_affine_predict", False)),
        local_affine_k=int(l.get("local_affine_k", 12)),
        local_affine_gain=float(l.get("local_affine_gain", 1.0)),
        local_affine_ridge=float(l.get("local_affine_ridge", 1.0)),
        # Per-track constant-velocity Kalman Mahalanobis gate (SOT-2920); absent keys
        # keep the champion byte-identical (kalman_gate default off). JSON cannot hold
        # inf, so an absent/null kalman_gate_chi2 means "no hard Mahalanobis rejection".
        kalman_gate=bool(l.get("kalman_gate", False)),
        kalman_process_noise=float(l.get("kalman_process_noise", 1.0)),
        kalman_obs_noise=float(l.get("kalman_obs_noise", 1.0)),
        kalman_gate_chi2=(
            float("inf")
            if l.get("kalman_gate_chi2") is None
            else float(l["kalman_gate_chi2"])
        ),
        kalman_init_pos_var=float(l.get("kalman_init_pos_var", 1.0)),
        kalman_init_vel_var=float(l.get("kalman_init_vel_var", 100.0)),
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
        # Whole-sequence Viterbi global track-linking with swaps (SOT-2918); absent
        # keys keep the champion byte-identical (viterbi_link default off). JSON cannot
        # hold inf, so an absent/null viterbi_theta means "unbounded" (accept every
        # distance-feasible pair => the birth/death arc never fires).
        viterbi_link=bool(l.get("viterbi_link", False)),
        viterbi_motion_gain=float(l.get("viterbi_motion_gain", 1.0)),
        viterbi_curvature_weight=float(l.get("viterbi_curvature_weight", 1.0)),
        viterbi_theta=(
            float("inf")
            if l.get("viterbi_theta") is None
            else float(l["viterbi_theta"])
        ),
        viterbi_max_sweeps=int(l.get("viterbi_max_sweeps", 8)),
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
        # Bidirectional mutual-NN cycle-consistency edge gate (SOT-2910); absent
        # keys keep the champion byte-identical (gate default off => no pruning).
        cycle_consistency_gate=bool(l.get("cycle_consistency_gate", False)),
        cycle_consistency_margin=float(l.get("cycle_consistency_margin", 0.0)),
        min_track_length=int(l.get("min_track_length", 1)),
        # The leading element is a ``kind`` tag; the trailing elements are the
        # kind's gate params (5 total for SOT-2818 ``nearest-head``, 7 for the
        # SOT-2898 ``mutual-nn`` precision fork). Preserve the FULL tuple rather
        # than truncating to 5 — the overlay dispatcher coerces each element by
        # position, so passing the raw config values through is both
        # backward-compatible (nearest-head sees the same 5) and forward-safe
        # (mutual-nn's require_primary_persist / mutual_margin survive).
        division_overlay=(
            (str(overlay[0]), *overlay[1:]) if overlay is not None else None
        ),
    )
    scale = tuple(config.get("scale", DEFAULT_SCALE))
    return detect, link, scale
