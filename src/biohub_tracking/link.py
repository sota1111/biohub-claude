"""Nearest-neighbour frame-to-frame linking with division handling.

Given per-timepoint centroids (from :mod:`biohub_tracking.detect`), build a
:class:`~biohub_tracking.graph.TrackingGraph`:

* **Assignment** — between consecutive timepoints ``t`` and ``t+1`` the scaled
  centroid distances form a cost matrix, solved with an optimal (minimum-cost)
  bipartite assignment (:func:`scipy.optimize.linear_sum_assignment`). Any pair
  farther apart than ``max_distance`` microns is rejected, so a cell that has no
  plausible successor simply ends its track (and a new detection starts one).
* **Division** — after the one-to-one assignment, each still-unmatched ``t+1``
  detection that lies within ``division_distance`` microns of an already-matched
  ``t`` parent is attached as that parent's *second* child, encoding a ``1 -> 2``
  split. A parent gains at most one extra child, so out-degree is capped at two,
  matching the competition's division definition. Two **over-split suppressors**
  keep spurious forks in check on dense volumes (SOT-2762): ``division_distance``
  can be tightened below ``max_distance``, and ``division_max_sibling_ratio``
  gates the second daughter on being roughly balanced with the primary daughter
  (a real mitotic split yields two daughters symmetric about the parent, so a
  detection much farther than the assigned child is not a plausible sibling).

The routine is deterministic and mirrors the competition's own micron-space,
optimal-assignment matching, so offline scores track the leaderboard metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .graph import TrackingGraph
from .matching import DEFAULT_MAX_DISTANCE, DEFAULT_SCALE


@dataclass(frozen=True)
class LinkParams:
    """Tunable knobs for :func:`link_centroids` (distances in microns)."""

    max_distance: float = DEFAULT_MAX_DISTANCE
    """Maximum scaled centroid distance for a frame-to-frame link."""

    allow_division: bool = True
    """Whether to attach a second daughter to a matched parent (``1 -> 2``)."""

    division_distance: float = DEFAULT_MAX_DISTANCE
    """Maximum parent->second-daughter distance when ``allow_division`` is set.

    An over-split suppressor: tightening this below :attr:`max_distance` restricts
    divisions to close, high-confidence sibling pairs, so a dense volume does not
    spray spurious forks (each of which is a division FP and, when its extra edge
    is evaluable, an edge FP too)."""

    division_max_sibling_ratio: float = 0.0
    """Sibling-balance over-split gate for divisions (SOT-2762).

    A real mitotic split produces two daughters roughly equidistant from the
    parent's last position. When ``> 0``, a leftover ``t+1`` detection is attached
    as a parent's second daughter only if its scaled parent-distance ``d2`` is
    ``<= division_max_sibling_ratio * d1``, where ``d1`` is the scaled distance
    from that parent to its *primary* (one-to-one assigned) daughter. This rejects
    attaching a distant unrelated detection as a fake sibling, the dominant
    over-split failure mode on the dense ``6bba`` families. ``0.0`` disables the
    gate (any leftover within ``division_distance`` may divide — the pre-SOT-2762
    behaviour, byte-for-byte)."""

    velocity_gain: float = 0.0
    """Constant-velocity motion prediction for the assignment (SOT-2369).

    A cell that already has an incoming edge (``t-1 -> t``) carries a velocity
    ``v = pos(t) - pos(t-1)``. When matching ``t -> t+1`` its predicted position
    becomes ``pos(t) + velocity_gain * v`` — a **damped** constant-velocity
    extrapolation ported from the public V40 lineage tracker's motion-aware
    reassignment (``q_hat = q + 0.5*(q - q_pred)``). The assignment then ranks
    candidate successors by distance to the *predicted* position, so a moving
    cell prefers the detection it is drifting toward rather than the closest
    stationary one. ``0.0`` disables prediction and reproduces the memoryless
    nearest-neighbour champion **byte-for-byte**."""

    velocity_disp_weight: float = 0.05
    """Tie-breaker weight on raw (unpredicted) displacement in the motion cost.

    Mirrors the ``0.05 * ||q_j - q_i||`` regulariser in the reference cost, so
    among motion-consistent candidates the assignment still slightly prefers the
    physically nearer detection. Only used when ``velocity_gain != 0``."""

    motion_gate_on_prediction: bool = False
    """Admissibility gate when motion prediction is active.

    ``False`` (default) keeps the feasibility gate on the **actual** scaled
    distance ``||pos(t) - pos(t+1)|| <= max_distance`` — motion only *re-ranks*
    within the champion's existing feasible set, so it can never introduce a new
    long-range edge (pure, low-risk improvement). ``True`` gates on the predicted
    distance instead, letting a fast, motion-consistent cell link slightly beyond
    ``max_distance`` in raw terms."""

    motion_model_link: bool = False
    """ARGUS-style motion-model predicted-position LAP linking (SOT-2864).

    ``False`` (default, absent key) reproduces the memoryless nearest-neighbour
    champion **byte-for-byte**. When set, the ``t -> t+1`` assignment is solved on
    **predicted** source positions taken from a *global, spatially-smoothed motion
    field* estimated from the current frame pair — the core of ARGUS
    (arXiv:2607.08297): match against *where each cell is predicted to move*, not
    its raw position (:func:`_motion_field_predict`).

    **Distinct mechanism** from the existing constant-velocity :attr:`velocity_gain`
    (SOT-2369): that predicts each cell from *its own* incoming ``t-1 -> t`` edge, so
    a first-appearance cell (no prior track) gets no prediction. The motion field is
    instead estimated *within the current frame pair* — a provisional assignment
    gives anchor displacements, which are smoothed over space (Gaussian kernel in
    scaled microns) into a dense field, so **every** source detection (including
    ones with no history) gets a locally-consistent predicted displacement. This is
    the portable, numpy/scipy-only proxy for a Farneback dense optical flow when
    ``cv2`` is unavailable (the offline kernel); the mechanism and gate reuse
    :attr:`velocity_disp_weight` / :attr:`motion_gate_on_prediction`. It is also
    mechanistically distinct from the rejected static gap-closing (SOT-2763) and
    node-interpolation gap-recovery (SOT-2849), which touch missing/gap frames
    rather than the primary ``t -> t+1`` link cost."""

    motion_smooth_sigma: float = 15.0
    """Gaussian bandwidth (scaled microns) of the motion-field smoothing (SOT-2864).

    Only used when :attr:`motion_model_link` is set. Each source detection's
    predicted displacement is the anchor displacements (from a provisional
    assignment) weighted by ``exp(-0.5 * (d / motion_smooth_sigma)**2)`` in scaled
    distance ``d`` to the anchor source. A larger sigma yields a smoother, more
    global field (approaching a single rigid translation); a smaller sigma lets the
    field vary locally. ``<= 0`` collapses to each cell's own provisional
    displacement (no spatial smoothing)."""

    motion_gain: float = 1.0
    """Scale on the predicted motion-field displacement (SOT-2864).

    Only used when :attr:`motion_model_link` is set. ``1.0`` uses the full predicted
    displacement; ``0.0`` disables prediction (falling back to the raw-position
    assignment); values in between damp the extrapolation."""

    local_affine_predict: bool = False
    """Local neighbour-affine motion prediction for the LAP link gate (SOT-2911).

    ``False`` (default, absent key) reproduces the memoryless nearest-neighbour
    champion **byte-for-byte**. When set, the ``t -> t+1`` assignment is solved on
    **predicted** source positions taken from a *per-cell local affine motion field*
    fitted by least squares to each cell's :attr:`local_affine_k` nearest anchor
    displacements (:func:`_local_affine_predict`), then reused by the existing
    predicted-position LAP path (``src_pred`` + :attr:`motion_gate_on_prediction`).

    **Distinct mechanism** from the SOT-2864 global smoothed motion field
    (:attr:`motion_model_link` / :attr:`motion_smooth_sigma`): that diffuses the
    provisional anchor displacements into a *single global* Gaussian-smoothed field —
    a locally-constant translation. This instead fits, per source cell, a *local
    affine* map ``disp ≈ A·pos + b`` (least squares over the k nearest anchors), so it
    captures the first-order **velocity gradient** of collective tissue flow (shear /
    divergence between adjacent morphogenetic patches — epiboly), which a smoothed
    average cannot represent. Ports the collective-motion affine/D²min analysis (PLOS
    Comput. Biol. pcbi.1008407 / PMC9287486) and Amat super-voxel optical flow
    (Bioinformatics 29(3):373). Pure numpy/scipy, CPU, offline, deterministic;
    evaluated as an **independent axis** (not combined with the global motion field).

    Both predicted-position paths reuse :attr:`velocity_disp_weight` /
    :attr:`motion_gate_on_prediction`; with ``motion_gate_on_prediction`` False
    (default) prediction only *re-ranks* within the champion's feasible set (the
    ``<= max_distance`` gate is on the raw distance), so it can never admit a new
    long-range edge."""

    local_affine_k: int = 12
    """Neighbour count for the local affine fit (SOT-2911).

    Only used when :attr:`local_affine_predict` is set. Each source cell fits its
    affine motion map to the ``local_affine_k`` nearest provisional anchors (by scaled
    distance). Fewer than 4 usable anchors falls back to the local mean displacement
    (a local translation); no anchor in reach keeps the cell at its raw position."""

    local_affine_gain: float = 1.0
    """Scale on the predicted local-affine displacement (SOT-2911).

    Only used when :attr:`local_affine_predict` is set. ``1.0`` uses the full fitted
    displacement; ``0.0`` disables prediction (raw-position assignment); values in
    between damp the extrapolation."""

    local_affine_ridge: float = 1.0
    """Ridge (Tikhonov) regularisation on the local affine fit (SOT-2911).

    Only used when :attr:`local_affine_predict` is set. Added to the normal-equation
    diagonal (in scaled-position units) to keep the per-cell least-squares affine well
    posed on nearly-collinear neighbourhoods; larger values shrink the fit toward the
    local mean translation."""

    kalman_gate: bool = False
    """Per-track constant-velocity Kalman state + Mahalanobis LAP gate (SOT-2920).

    ``False`` (default, absent key) reproduces the memoryless nearest-neighbour /
    motion-field champion **byte-for-byte**. When set, the ``t -> t+1`` linking runs a
    dedicated sequential path (:func:`_kalman_link`) that keeps, **per track**, a
    constant-velocity Kalman filter (state ``[z, y, x, vz, vy, vx]`` in scaled microns).
    Each track predicts its next position ``mu = H·F·state`` and the *innovation
    covariance* ``S = H·(F·P·Fᵀ + Q)·Hᵀ + R``; the assignment cost is the
    **Mahalanobis** distance ``d_M(z, mu, S) = sqrt((z-mu)ᵀ S⁻¹ (z-mu))`` and the gate is
    ``d_M² <= :attr:`kalman_gate_chi2``` **intersected** with the champion's raw
    ``<= :attr:`max_distance``` Euclidean cap (so the gate can only *tighten/re-weight*
    the champion feasible set — never admit a new long-range, metric-invalid edge).

    **Adaptivity — the genuinely new signal.** The 7 µm Euclidean gate is a single
    fixed radius for every track. Here the gate is normalised by each track's own
    innovation covariance ``S``: a track with a well-established velocity has small ``S``
    and so tightly prefers the detection it is predicted to drift toward (rejecting a
    merely-near distractor), while a fresh / history-less track has an inflated ``S``
    (:attr:`kalman_init_vel_var`) and so gates ~isotropically like the Euclidean champion.

    **Distinct mechanism** from every prior linking axis (evaluated as a **single axis**,
    with :attr:`motion_model_link` **OFF**, so its contribution is separated from the
    champion's global smoothed motion field): the SOT-2864 field
    (:attr:`motion_model_link`) diffuses a *single global* Gaussian-smoothed translation
    estimated within the frame pair, and the rejected SOT-2911 local-affine
    (:attr:`local_affine_predict`) fits a spatial *velocity gradient* from neighbour
    displacements — **both are spatial, history-free** fields shared across cells. This
    instead carries each track's **own temporal (z,y,x,v) state** across frames and, uniquely
    among these axes, forms an *anisotropic, uncertainty-normalised* Mahalanobis gate from
    the propagated covariance rather than a fixed Euclidean radius. It is also richer than
    the memoryless-cell :attr:`velocity_gain` damped extrapolation (SOT-2369), which has no
    covariance and no per-track gate. Pure numpy/scipy, CPU, offline, deterministic."""

    kalman_process_noise: float = 1.0
    """Process-noise scale ``q`` (scaled microns) of the CV Kalman filter (SOT-2920).

    Only used when :attr:`kalman_gate` is set. Builds the discrete white-noise-
    acceleration ``Q = q² · [[¼I, ½I],[½I, I]]`` (dt=1). Larger ``q`` inflates the
    predicted covariance ``S`` → a wider, softer Mahalanobis gate that trusts the
    constant-velocity prediction less (approaching the Euclidean champion); smaller ``q``
    trusts the motion model more (a tighter, more selective gate)."""

    kalman_obs_noise: float = 1.0
    """Observation-noise std ``r`` (scaled microns) of the CV Kalman filter (SOT-2920).

    Only used when :attr:`kalman_gate` is set. Sets the measurement covariance
    ``R = r²·I`` added to the innovation covariance ``S`` and used in the Kalman update
    gain. Larger ``r`` down-weights each detection (smoother velocity estimate, wider
    gate); must be ``> 0`` so ``S`` is always positive-definite / invertible."""

    kalman_gate_chi2: float = float("inf")
    """Mahalanobis² gate threshold (χ², 3 DOF) for the CV Kalman path (SOT-2920).

    Only consulted when :attr:`kalman_gate` is set. A ``t -> t+1`` pair is admitted only
    when its squared Mahalanobis distance ``d_M² <= kalman_gate_chi2`` **and** its raw
    scaled distance ``<= :attr:`max_distance``` (the intersection keeps the champion's
    hard 7 µm cap, so this can only restrict/re-rank, never admit a long-range edge).
    ``inf`` (default) applies no hard Mahalanobis rejection — the covariance only
    *re-weights* the cost within the Euclidean feasible set. Finite values near the χ²
    quantiles (e.g. ``7.815`` = 0.95, ``11.345`` = 0.99) additionally drop
    motion-inconsistent (FP-prone) successors."""

    kalman_init_pos_var: float = 1.0
    """Initial position variance (scaled microns²) for a fresh track's Kalman state
    (SOT-2920). Only used when :attr:`kalman_gate` is set."""

    kalman_init_vel_var: float = 100.0
    """Initial *velocity* variance (scaled microns²) for a fresh track's Kalman state
    (SOT-2920). Only used when :attr:`kalman_gate` is set. Large by default so a
    history-less track's first-step innovation covariance is dominated by position/obs
    noise and its Mahalanobis gate is ~isotropic (≈ the Euclidean champion) until the
    filter has seen enough steps to estimate a confident velocity."""

    link_consistency_gate: bool = False
    """Ultrack-style bidirectional forward↔backward motion-consistency gate (SOT-2883).

    ``False`` (default, absent key) reproduces the champion / SOT-2864 motion linker
    **byte-for-byte**. When ``True`` (and :attr:`motion_model_link` is set — the gate
    reuses the SOT-2864 global smoothed motion field, so it is inert without it), a
    ``t -> t+1`` link ``i -> j`` is judged by *mutual* prediction agreement, the
    portable point-detection surrogate for Ultrack's (Nature Methods 2025,
    arXiv:2308.04526) adjacent-frame **overlap-maximization** selection principle:

    * **forward residual** ``r_f`` — the SOT-2864 *forward* motion field predicts where
      ``src[i]`` moves; ``r_f = || src_pred_fwd[i] - dst[j] ||`` (scaled microns).
    * **backward residual** ``r_b`` — the SAME field estimated on the *reversed* frame
      pair predicts where ``dst[j]`` came from; ``r_b = || dst_pred_bwd[j] - src[i] ||``.

    A genuine cell has BOTH small (its forward flow lands on the target and the
    target's backward flow lands back on it); an FP-prone link has a large residual in
    at least one direction (the forward field can over-smooth a spurious near
    neighbour, but the backward field disagrees). The gate therefore:

    * **penalises/discounts** — adds :attr:`link_consistency_weight` ``* 0.5*(r_f+r_b)``
      to the assignment cost, so among distance-feasible candidates a *bidirectionally*
      consistent successor is preferred; and
    * **rejects** — when :attr:`link_consistency_tol` is finite, drops any pair with
      ``r_f > tol`` OR ``r_b > tol`` from the feasible set.

    It is a pure **restriction** of the SOT-2864 feasible set (never admits a new pair),
    so the ``<= max_distance`` cap is preserved by construction — a wild motion
    prediction can never let it link an arbitrarily distant pair. **Mechanistically
    distinct** from the rejected SOT-2871 windowed running-average velocity (a *carried*
    single-direction trajectory, not forward↔backward agreement) and SOT-2870 learned
    edge-gate (a *learned* one-direction p_edge, not a symmetric field cross-check)."""

    link_consistency_tol: float = float("inf")
    """Hard-gate tolerance (scaled microns) on the bidirectional residuals (SOT-2883).

    Only consulted when :attr:`link_consistency_gate` is set. A ``t -> t+1`` pair is
    rejected when EITHER its forward residual ``r_f`` or backward residual ``r_b``
    exceeds this. ``inf`` (default) applies no hard rejection (soft penalty only);
    tightening it below :attr:`max_distance` removes the motion-inconsistent (FP-prone)
    links whose two direction predictions disagree."""

    link_consistency_weight: float = 0.0
    """Soft-penalty weight on the mean bidirectional residual (SOT-2883).

    Only consulted when :attr:`link_consistency_gate` is set. Adds
    ``link_consistency_weight * 0.5 * (r_f + r_b)`` to the assignment cost, so a
    bidirectionally consistent successor is preferred among the distance-feasible
    candidates. ``0.0`` (default) adds no penalty (hard gate only, if any)."""

    max_frame_gap: int = 1
    """Gap-closing 2nd linking step across missing detections (SOT-2763).

    The frame-to-frame assignment above only links consecutive frames, so a cell
    that is *missed* in one frame ends its track and its persistence is lost —
    the dominant source of FN edges. This is the classical Jaqaman/TrackMate
    second LAP step: after frame-to-frame linking, each track *fragment* terminal
    (a **tail** = node with no successor) is bridged to a later fragment **head**
    (node with no predecessor) whose timepoint is ``t + g`` for a frame gap
    ``2 <= g <= max_frame_gap``, within :attr:`gap_distance` microns, by an
    optimal min-cost assignment (:func:`_gap_close`). ``1`` disables gap-closing
    and reproduces the frame-to-frame champion **byte-for-byte**.

    **Interaction with the metric (important).** The competition edge metric keeps
    only consecutive-frame edges (``t_target - t_source == 1``), so a bridge edge
    that spans a gap is *dropped* before scoring — it is neither TP nor FP. Its
    entire value is therefore indirect: gap-closing runs **before**
    :attr:`min_track_length` pruning, so bridging two real short fragments into one
    ``>= min_track_length``-node weakly-connected component rescues their internal
    *consecutive* edges from being pruned (recovering FN edges), at the cost of
    keeping the bridged nodes (raising ``N_pred`` and the node-count penalty). The
    net effect is decided empirically on the leak-free CV."""

    gap_distance: float = DEFAULT_MAX_DISTANCE
    """Maximum tail->head scaled distance (microns) for a gap-closing bridge.

    Only used when :attr:`max_frame_gap` ``> 1``. A single absolute gate (the
    TrackMate "gap-closing max distance" convention) rather than a per-frame
    budget, so widening it admits more distant reconnections; tightening it keeps
    only close, high-confidence bridges to suppress spurious cross-track merges."""

    gap_recover: bool = False
    """Node-interpolation gap recovery on missing detection frames (SOT-2849).

    ``False`` (default) reproduces the champion graph **byte-for-byte**. When set,
    after frame-to-frame linking, a track-fragment **tail** at ``t`` is reconnected
    to a later fragment **head** at ``t + g`` (``2 <= g <= gap_recover_max_gap``,
    within :attr:`gap_recover_distance` microns) by inserting **interpolated
    detection nodes** at each missing frame ``t+1 .. t+g-1`` (linear interpolation
    of the voxel centroid between tail and head) wired as a chain of *consecutive*
    edges ``tail -> interp -> ... -> head`` (:func:`_gap_recover`).

    **Why this is distinct from gap-closing (SOT-2763).** ``max_frame_gap`` added a
    single *non-consecutive* bridge edge, which the competition edge metric drops
    (``t_target - t_source == 1`` only) — so that mechanism could never recover an
    FN edge and was rejected. Node interpolation instead produces only consecutive
    edges, which ARE scored; an interpolated node landing within the ``<= 7 µm``
    per-timepoint match radius of the true (missed-detection) GT node recovers a
    real FN edge (the ``pilkwang`` "gap recovery" mechanism). Runs **before**
    :attr:`min_track_length` pruning, so a recovered bridge can also lift two real
    short fragments into one surviving ``>= min_track_length`` component. The net
    effect (FN recovery vs. interpolated-node FP / node-count penalty) is decided
    empirically on the leak-free CV."""

    gap_recover_max_gap: int = 2
    """Maximum frame gap a :attr:`gap_recover` bridge may span (``>= 2``).

    ``g`` missing frames insert ``g - 1`` interpolated nodes. Only used when
    :attr:`gap_recover` is set; ``2`` bridges a single missing frame."""

    gap_recover_distance: float = DEFAULT_MAX_DISTANCE
    """Maximum tail->head scaled distance (microns) for a :attr:`gap_recover` bridge.

    A single absolute gate (the TrackMate gap-closing-max-distance convention).
    Only used when :attr:`gap_recover` is set."""

    gap_recover_min_frag: int = 1
    """Minimum weakly-connected fragment size at each :attr:`gap_recover` terminal.

    Both the tail's and the head's fragment must contain ``>= gap_recover_min_frag``
    nodes to be eligible for a bridge. ``1`` (default) admits any terminal; a larger
    value refuses to reconnect (and resurrect through the prune) the short noise
    fragments the champion's :attr:`min_track_length` prune is designed to remove —
    the node-resurrection failure mode that sank gap-closing (SOT-2763). Only used
    when :attr:`gap_recover` is set."""

    division_overlay: tuple | None = None
    """Non-destructive division-event overlay (SOT-2818), applied *after* linking.

    ``None`` (default) disables the overlay and reproduces the champion graph
    **byte-for-byte**. When set it is
    ``(kind, max_distance, sibling_ratio, min_daughter_len, require_parent_track)``
    (see :mod:`biohub_tracking.division_overlay`): a pure post-processing pass that
    re-attaches the *second* daughter the one-to-one linker dropped at a real
    division — adding ``parent -> nearby-persistent-head`` edges only, never moving
    the primary assignment, adding a node, or deleting an edge. It runs last (after
    :attr:`min_track_length` pruning) so daughter-lineage persistence is measured on
    the final graph. It is the score lever that recovers the metric's ``0.1 ·
    division_jaccard`` term without the LAP reassignment that made in-linker
    ``allow_division`` (SOT-2762) regress the edge metric."""

    global_window: int = 1
    """Short-window global min-cost-flow assignment with explicit birth/death arcs
    (SOT-2830, portable tracking-by-assignment; arxiv 2004.06375 / 1705.03386).

    ``1`` (default) runs the **unchanged per-frame path** above (memoryless
    optimal bipartite matching per ``t -> t+1`` transition), so the champion graph
    is reproduced **byte-for-byte**. ``>= 2`` activates the global assignment path
    (:func:`_global_link`): the same ``t -> t+1`` *metric-valid* candidate links
    (never a non-consecutive **bridge** edge — that is what made gap-closing
    (SOT-2763) forfeit the non-continuous edge metric), but each detection also
    gets explicit **birth** and **death** arcs (:attr:`birth_cost` /
    :attr:`death_cost`) so the solver may *refuse* a link and start/end a track
    instead of greedily attaching every feasible pair. This is the classical
    network-flow tracker (Zhang 2008) restricted to a short temporal window and to
    adjacent-frame arcs.

    **Reduction (why the window is a structural knob, not a second lever).** With
    the champion's pure-distance edge cost and consecutive-only arcs, the window
    min-cost flow *decouples per transition*: a middle detection's incoming and
    outgoing links share no flow variable (the birth/death costs are per-node
    constants, so each transition's assignment-with-outliers is independent). So
    for this cost model the joint W-frame optimum equals the per-transition
    optimum, and :func:`_global_link` solves it exactly and efficiently as one
    assignment-with-birth/death-outliers per transition (:func:`_global_assign`);
    ``global_window`` only *activates* the birth/death arcs. The value is reserved
    for a future cross-hop coupling term (velocity/appearance) that would make the
    window bite; today the effective lever is the birth/death threshold
    ``theta = birth_cost + death_cost``. ``global_window >= 2`` is incompatible with
    in-linker ``allow_division`` / gap-closing (``max_frame_gap > 1``); those knobs
    are ignored on the global path so it emits only ``t -> t+1`` one-to-one edges."""

    birth_cost: float = float("inf")
    """Cost of a detection *starting* a track (no incoming ``t-1 -> t`` link) on the
    global path (:attr:`global_window` ``>= 2``). ``inf`` (default) makes every
    feasible link strictly cheaper than a birth+death, so the global path matches
    **all** feasible pairs exactly like the per-frame champion. A finite value
    forms the link-acceptance threshold ``theta = birth_cost + death_cost``: a
    ``t -> t+1`` pair is linked only when its scaled distance is ``< theta`` (and
    ``<= max_distance``), so lowering ``theta`` suppresses the marginal
    long-distance mis-links a greedy per-frame assignment makes to transient
    detections. Ignored when :attr:`global_window` ``<= 1``."""

    death_cost: float = float("inf")
    """Cost of a detection *ending* a track (no outgoing ``t -> t+1`` link) on the
    global path. See :attr:`birth_cost`; the two enter the assignment only through
    their sum ``theta``. Ignored when :attr:`global_window` ``<= 1``."""

    appearance_weight: float = 0.0
    """Local appearance-descriptor similarity term in the frame-to-frame link cost
    (SOT-2829, portable analog of the official cross-attention appearance linker).

    The champion links ``t -> t+1`` on **scaled centroid distance alone**, so in a
    dense family (``6bba``) where several plausible successors sit within
    ``max_distance`` the optimal-distance assignment can attach the wrong, merely
    nearer neighbour. When ``> 0`` the assignment cost becomes
    ``dist + appearance_weight * (1 - similarity)`` where ``similarity`` is the
    cosine of the two detections' standardised local appearance descriptors
    (:func:`biohub_tracking.detect.patch_descriptors`) mapped to ``[0, 1]``, so a
    successor that *looks* like the source is preferred among the distance-feasible
    candidates. The ``<= max_distance`` feasibility gate stays on the **raw scaled
    distance**, so appearance only *re-ranks* the champion's existing feasible set
    and can never introduce a new long-range (metric-invalid) edge — a pure,
    low-risk disambiguation. ``0.0`` (default) drops the term entirely and
    reproduces the distance-only champion **byte-for-byte**; it is also inert when
    no descriptors are supplied to :func:`link_centroids`."""

    edge_cost_model: dict | None = None
    """GT-learned edge-linking cost, embedded logistic model (SOT-2841, default-off).

    The single hand-crafted :attr:`appearance_weight` term (SOT-2829) was a lone,
    non-learned feature; this is the untried **learned LINKING** axis (``label =
    edge``): a light logistic classifier fit **leak-free (leave-one-family-out)** on
    GT *consecutive* edges vs. the feasible non-GT successors of matched sources,
    over a joint edge-feature vector (scaled distance, appearance-descriptor cosine,
    and dense-cluster competition cues — rival counts, distance ranks, distance
    margin; see :data:`biohub_tracking.edge_linker.EDGE_FEATURE_NAMES`). Its learned
    probability ``p_edge`` enters the assignment cost as ``dist +
    edge_cost_model.weight * (1 - p_edge)``, so a successor the classifier judges a
    genuine GT link is preferred among the distance-feasible candidates.

    The dict is the serialized :class:`biohub_tracking.edge_linker.LearnedEdgeCost`
    (``feature_names``/``mean``/``std``/``coef``/``intercept``/``weight``). As with
    :attr:`appearance_weight`, the ``<= max_distance`` feasibility gate stays on the
    **raw scaled distance**, so the term only *re-ranks* the champion's existing
    feasible set and can never introduce a new long-range (metric-invalid) edge —
    pure, low-risk disambiguation. ``None`` (default), a model ``weight == 0``, or no
    supplied ``descriptors`` all drop the term and reproduce the distance-only
    champion **byte-for-byte** (a strict default-off superset). Inactive on the
    global-window (SOT-2830) path (which emits distance-only birth/death edges)."""

    edge_gate_expand: bool = False
    """Learned-cost feasibility-gate EXPANSION for FN-edge recovery (SOT-2870, off).

    SOT-2841's re-rank was CV-neutral (a fixed raw-distance feasible set is
    saturated); SOT-2864 showed the headroom is in the *gate* — admitting a raw-far
    but **motion-consistent** successor the distance gate drops (an FN edge). When
    this is ``True`` (and an :attr:`edge_cost_model` and a motion prediction —
    :attr:`motion_model_link` or :attr:`velocity_gain` — and descriptors are all
    active), a pair whose **raw** scaled distance exceeds :attr:`max_distance` is
    additionally admitted iff ALL hold:

    * it is **motion-corrected in-range**: the SOT-2864 predicted-position distance is
      ``<= max_distance`` (so the expansion is confined to motion-explained proximity,
      never an unbounded long-range edge);
    * its raw distance is within :attr:`edge_gate_expand_ratio` ``* max_distance`` (a
      hard cap so a wild motion prediction cannot admit an arbitrarily distant pair);
    * the learned ``p_edge`` (motion + shape/intensity joint features) is
      ``>= :attr:`edge_gate_admit_prob``` — the classifier must judge it a real edge.

    ``False`` (default) leaves the feasibility gate on the raw distance exactly like
    SOT-2841 (re-rank only), so the champion is reproduced **byte-for-byte**."""

    edge_gate_admit_prob: float = 0.5
    """Min learned ``p_edge`` to admit a gate-expanded pair (SOT-2870).

    Only consulted when :attr:`edge_gate_expand` is set. Higher = stricter (fewer
    expanded admits, fewer FP; the learned filter that SOT-2864's pure motion gate
    lacked)."""

    edge_gate_expand_ratio: float = 1.5
    """Hard cap on a gate-expanded pair's **raw** distance, as a multiple of
    :attr:`max_distance` (SOT-2870). Bounds the expansion so a spurious motion
    prediction can never admit an arbitrarily far (metric-invalid) edge."""

    xattn_edge_model: dict | None = None
    """Learned CROSS-ATTENTION edge-linking cost (SOT-2994, SimpleNodeTransformer port).

    The embedded-coefficient config block of a
    :class:`biohub_tracking.xattn_edge.CrossAttentionEdgeCost`. Unlike the per-edge
    logistic :attr:`edge_cost_model` (SOT-2841, REJECTED), each edge embedding is
    contextualised by attention over its source's competing successors and its
    destination's competing predecessors before being scored, so the assignment sees
    the whole candidate set (the official ``SimpleNodeTransformer`` idea) rather than
    one pair in isolation. Its ``weight * (1 - p_edge)`` penalty is **added to the
    cost only** exactly like :attr:`edge_cost_model` — the ``<= max_distance``
    feasibility gate stays on the raw/motion distance, so it re-ranks the champion's
    feasible set and never admits an out-of-range edge (metric-valid). Inference is
    pure numpy (exec-compat). ``None`` (default) or an embedded ``weight == 0`` or no
    descriptors drops the term and reproduces the champion **byte-for-byte**."""

    window_assoc: int = 1
    """Portable Trackastra-style windowed global association (SOT-2871, default-off).

    ``1`` (default, absent key) runs the **unchanged per-frame champion path**, so
    the champion graph is reproduced **byte-for-byte**. ``>= 2`` activates a
    *motion-coupled windowed LAP chain* (:func:`_window_link`): the ``t -> t+1``
    assignment is solved on **motion-predicted** source positions whose predicted
    displacement blends the SOT-2864 global motion field with a **carried velocity**
    running-averaged over the previous ``window_assoc - 1`` transitions of the
    window. That carried velocity is what makes the window *bite* — a middle
    detection's outgoing link cost depends on the incoming link the window chose one
    frame earlier (a cross-hop coupling the SOT-2830 pure-distance min-cost-flow
    provably lacks, so its window decoupled). It is Trackastra's (arXiv:2405.15700)
    short sliding-window association ported portably: **numpy/scipy only** (a LAP
    chain with birth/death outlier arcs — :func:`_window_assign`), no torch /
    attention / pretrained weights / cv2.

    **Distinct from prior linking axes.** Unlike static gap-closing (SOT-2763,
    rejected: the metric drops the non-consecutive bridge edge) and node-interp
    gap-recovery (SOT-2849, rejected: family-mix sensitive), this touches only the
    primary consecutive ``t -> t+1`` link and emits only metric-scored consecutive
    edges. Unlike the SOT-2830 global min-cost-flow (``global_window``, +0.0022 and
    family-mix sensitive — its cost decoupled per transition), the motion carry
    couples adjacent transitions so the window is a genuine joint reasoning step.
    Unlike the memoryless-cell ``velocity_gain`` (SOT-2369), the carried velocity is
    re-derived from *the window's own chosen links* and blended with the global
    field, so even a history-less cell inherits neighbourhood motion.

    ``window_assoc >= 2`` supersedes / is incompatible with ``global_window`` (they
    are mutually exclusive global paths); in-linker division on this path is governed
    by :attr:`window_parental_softmax`, and gap-closing knobs are ignored (only
    consecutive edges are emitted)."""

    window_theta: float = float("inf")
    """Link-acceptance threshold on the windowed path (SOT-2871). Birth+death outlier
    cost sum: a ``t -> t+1`` pair is linked only when its **effective** (motion-
    predicted + optional appearance/edge) cost is ``< window_theta`` (and its gate
    distance is ``<= max_distance``). ``inf`` (default) accepts every feasible pair,
    so with motion off the windowed path reproduces the per-frame champion matching
    exactly. Lowering it lets the solver *refuse* a marginal link and start/end a
    track instead (the classical network-flow outlier). Ignored when
    :attr:`window_assoc` ``<= 1``."""

    window_carry_weight: float = 0.5
    """Blend weight on the carried window velocity vs the SOT-2864 global motion
    field in the windowed predicted displacement (SOT-2871). For a source with a
    window history the predicted displacement is ``window_carry_weight * carried +
    (1 - window_carry_weight) * field``; a history-less source uses the field alone.
    ``0.0`` collapses to the pure SOT-2864 motion field (the window stops biting);
    ``1.0`` trusts the carried trajectory only. Ignored when :attr:`window_assoc`
    ``<= 1`` or :attr:`motion_gain` ``== 0``."""

    window_parental_softmax: bool = False
    """Parental-softmax division constraint within the window (SOT-2871, default-off).

    Trackastra's parental softmax normalises each parent's association mass over its
    candidate children so a parent's total child-association is ``<= 1`` (one parent,
    possibly two children, never a spurious spray of forks). Here, when
    :attr:`allow_division` is set, a leftover ``t+1`` detection is attached to a
    matched parent as a **second daughter** only if its softmax association share
    (``softmax(-scaled_dist / window_softmax_temp)`` over that parent's feasible
    children within :attr:`division_distance`) is ``>= window_softmax_min_share`` —
    so the parent's mass is genuinely *shared* between two comparable daughters (a
    balanced mitotic split), not leaked to a distant unrelated detection (a division
    FP). ``False`` (default) or :attr:`allow_division` off adds no fork. Only used on
    the :attr:`window_assoc` ``>= 2`` path."""

    window_softmax_min_share: float = 0.3
    """Minimum parental-softmax association share to admit a second daughter
    (SOT-2871). Higher = stricter (a candidate sibling must carry more of the
    parent's normalised association mass, suppressing more division FPs). Only used
    when :attr:`window_parental_softmax` and :attr:`allow_division` are set."""

    window_softmax_temp: float = 1.0
    """Softmax temperature (scaled microns) for the parental-softmax shares
    (SOT-2871). Smaller = sharper (mass concentrates on the nearest child, so a
    second daughter rarely clears :attr:`window_softmax_min_share`); larger =
    flatter. Only used when :attr:`window_parental_softmax` is set."""

    suspicious_review: bool = False
    """Post-hoc suspicious-tracking-event review gate (SOT-2895, default-off).

    Ported from the public **"Biohub Suspicious Tracking Event Review"** notebook
    (dalloliogm, Apache-2.0,
    https://www.kaggle.com/code/dalloliogm/biohub-suspicious-tracking-event-review),
    which builds *graph diagnostics* over a submitted lineage graph to surface
    anomalous tracking events. Here that review is applied as a **post-hoc,
    track-unit anomaly gate** on the already-linked graph: an interior link is cut
    when it is simultaneously (a) a sharp **direction reversal** and (b) an
    abnormal **step jump** relative to the cell's own incoming motion — the
    teleport-to-decoy signature of a linking false-positive that grabbed a
    transient/noise detection. Cutting such an edge ends the real track cleanly and
    leaves the decoy tail as a short fragment that the existing
    :attr:`min_track_length` prune (which runs *after* this gate) removes, freeing
    the matched GT node and shedding a false-positive edge — the score lever, since
    the metric is ``TP/(TP+FP+FN)`` and FP edges hurt most.

    ``False`` (default, absent key) leaves the champion graph **byte-for-byte**
    unchanged. This is a *post-hoc, per-track anomaly detection* family, distinct
    from every rejected linking axis, none of which inspect the finished track's
    self-consistency: static gap-closing (SOT-2763), GT-learned edge cost
    (SOT-2841), learned motion/shape edge gate (SOT-2870) and the bidirectional
    forward↔backward link-cost gate (SOT-2883) all reshape the ``t -> t+1``
    *assignment cost*, whereas this only removes an already-made edge that the
    finished trajectory contradicts."""

    suspicious_turn_cos: float = -0.5
    """Direction-reversal threshold for the suspicious-review gate (SOT-2895).

    At an interior node ``u`` with incoming displacement ``d1 = pos(u) - pos(p)``
    and outgoing displacement ``d2 = pos(v) - pos(u)`` (both in **scaled microns**),
    the turn cosine is ``cos = (d1·d2)/(|d1||d2|)``. The outgoing edge is a
    reversal candidate when ``cos < suspicious_turn_cos``. The default ``-0.5``
    requires a turn sharper than 120° — a near-backtrack a smoothly moving cell
    almost never makes. Only used when :attr:`suspicious_review` is set."""

    suspicious_jump_ratio: float = 3.0
    """Step-jump threshold for the suspicious-review gate (SOT-2895).

    The outgoing edge is a jump candidate when its scaled step ``|d2|`` exceeds
    ``suspicious_jump_ratio * max(|d1|, suspicious_jump_floor)`` — the cell
    suddenly accelerates far beyond its established pace. An edge is cut only when
    it is **both** a reversal (:attr:`suspicious_turn_cos`) **and** a jump, so a
    fast-but-straight cell (jump, no reversal) and a gentle wobble (reversal, no
    jump) are both preserved. Higher = stricter (fewer cuts). Only used when
    :attr:`suspicious_review` is set."""

    suspicious_jump_floor: float = 1.0
    """Minimum reference step (scaled microns) in the jump test (SOT-2895).

    Guards the ``|d2| > ratio * |d1|`` test against a near-static cell whose tiny
    ``|d1|`` would make any real step look like a huge multiple. The reference step
    is ``max(|d1|, suspicious_jump_floor)``. Only used when
    :attr:`suspicious_review` is set."""

    link_two_pass: bool = False
    """Two-pass tight-then-full-gate Hungarian linking (SOT-2899, default-off).

    Ported from the genuine public **classical baseline** (xiaoleilian, LB 0.720,
    https://www.kaggle.com/code/xiaoleilian/biohub-cell-tracking-classical-baseline,
    which links "v4: two-pass — tight gate first, then full gate for leftovers"):
    a single global optimal assignment commits early to a *distractor steal* — in a
    dense volume it may assign a source to a merely-near wrong neighbour, then have
    no successor left for the cell that truly moved there. The two-pass structure
    instead solves the assignment in **two stages**:

    * **Pass 1 (tight gate).** An optimal one-to-one assignment within the champion's
      :attr:`max_distance` (the tight gate — cells move ``<= 7 µm`` at p99 while
      neighbours are ``>= 12 µm`` apart, so this pass makes only high-confidence
      links and cannot steal across the neighbour spacing).
    * **Pass 2 (full gate).** The still-**unmatched** sources and destinations only
      are re-linked with an optimal assignment out to :attr:`link_full_distance`
      (``>= max_distance``), recovering the rare fast-moving cell whose real
      successor sits just beyond the tight gate (an FN edge) — without ever letting
      the wider gate perturb a Pass-1 high-confidence link.

    This is a pure **feasibility-structure** change to the primary ``t -> t+1``
    assignment, mechanistically distinct from every rejected linking axis: it is
    not a cost re-rank (appearance SOT-2829 / learned-edge SOT-2841), not a
    cross-frame bridge (gap-closing SOT-2763 / gap-recover SOT-2849), not a global
    birth/death flow (SOT-2830), not a post-hoc cut (suspicious-review SOT-2895),
    and not motion prediction (SOT-2864). ``False`` (default), or
    ``link_full_distance <= max_distance`` (Pass 2 is then a no-op), reproduces the
    memoryless single-pass champion **byte-for-byte**. Active only on the per-frame
    champion path (ignored when a motion / appearance / learned-edge / windowed /
    global-flow lever is engaged, whose own feasible sets already differ)."""

    link_full_distance: float = DEFAULT_MAX_DISTANCE
    """Pass-2 (leftover-recovery) gate for :attr:`link_two_pass`, in microns.

    Only used when :attr:`link_two_pass` is set. The second assignment re-links the
    Pass-1 leftovers out to this scaled distance; it must be ``> max_distance`` to
    have any effect (``<= max_distance`` makes Pass 2 a strict no-op, reproducing
    the champion byte-for-byte). The classical baseline used a full gate ``~1.3×`` its
    tight gate (11 µm over 7 µm)."""

    cycle_consistency_gate: bool = False
    """Bidirectional mutual-nearest-neighbour cycle-consistency edge gate (SOT-2910).

    ``False`` (default, absent key) leaves the champion assignment **byte-for-byte**
    unchanged. When ``True``, the primary ``t -> t+1`` links are additionally filtered
    to keep only those that are **mutually consistent in both directions** — a pure
    logic FP-edge suppressor ported from the cycle-consistency principle of
    NeighborTrack (arXiv:2211.06663) and DistNet2D (arXiv:2310.19641):

    * **forward** — for source ``i`` its nearest destination is ``argmin_j dist(i, j)``
      (scaled microns);
    * **backward** — for destination ``j`` its nearest source is ``argmin_i dist(i, j)``.

    A link ``i -> j`` survives only when it is a **mutual nearest neighbour**: ``j`` is
    ``i``'s forward-nearest destination AND ``i`` is ``j``'s backward-nearest source.
    The champion's optimal (Hungarian) assignment sacrifices local mutual-nearest
    optimality for a globally-cheaper total; in a dense volume that produces exactly
    the *contested*, non-mutual links most likely to be a false-positive steal. Dropping
    an asymmetric link ends its source's track cleanly and leaves a shorter fragment the
    existing :attr:`min_track_length` prune (which runs after) removes — shedding the FP
    edge, the score lever since the metric is ``TP/(TP+FP+FN)``.

    This **PRUNES** primary ``t -> t+1`` edges and never adds one, so it can only lose
    recall to gain precision (never introduces a new/long-range edge); the ``<=
    max_distance`` feasibility cap is preserved by construction. It is a strict
    **restriction** of the champion feasible set. **Distinct** from the REJECTED
    SOT-2895 suspicious-review (a *jump/reversal* self-motion signature on the finished
    track, non-specific), the REJECTED SOT-2898 ``mutual-nn`` **division** overlay (which
    *added* second-daughter fork edges to realise the 0.1 division term — this axis adds
    none), and the SOT-2883 forward↔backward *motion-field residual* gate (which reuses
    the SOT-2864 smoothed motion field and only re-ranks/soft-penalises within it — this
    axis is a pure point mutual-NN filter that needs no motion model). Active on the
    per-frame champion path (the global/window paths take their own early return)."""

    cycle_consistency_margin: float = 0.0
    """Ambiguity margin (scaled microns) for the cycle-consistency gate (SOT-2910).

    Only consulted when :attr:`cycle_consistency_gate` is set. ``0.0`` (default) keeps
    the pure mutual-nearest-neighbour rule (drop every non-mutual link). When ``> 0`` a
    surviving mutual-best link ``i -> j`` is *additionally* dropped when it is
    **contested** — its runner-up competitor is within ``margin`` in **either**
    direction (``second_nearest_dst(i) - dist(i,j) < margin`` OR
    ``second_nearest_src(j) - dist(i,j) < margin``). Larger ``margin`` = stricter (only
    unambiguous, well-separated mutual links survive), suppressing more FP edges at the
    cost of more recall. A source/destination with no competitor is never contested."""

    viterbi_link: bool = False
    """Whole-sequence Viterbi global track-linking with swap operations (SOT-2918).

    ``False`` (default, absent key) runs the unchanged per-frame champion path, so
    the champion graph is reproduced **byte-for-byte**. When ``True`` the linker
    takes an early return through :func:`_viterbi_link`, a portable numpy/scipy port
    of Magnusson/Jaldén/Gilbert/Blau, *"Global Linking of Cell Tracks Using the
    Viterbi Algorithm"* (IEEE TMI 34(4):911-929, 2015; the ISBI Cell-Tracking-
    Challenge-winning global linker).

    **Mechanism (whole-sequence, with swaps).** The champion is a *greedy* per-frame
    Hungarian LAP: a link committed at ``t -> t+1`` can never be revised, so a single
    early mis-link *propagates* down the track. Magnusson's linker instead scores
    each ``t -> t+1`` assignment against a *motion-coherence* trellis coupling both
    neighbouring transitions — the predicted position of the source from its
    **incoming** velocity (``t-1 -> t``) *and* the predicted origin of the
    destination from its **outgoing** velocity (``t+1 -> t+2``) — and re-solves the
    **whole sequence** by iterated re-optimisation (:func:`_viterbi_link`) until the
    assignment reaches a fixed point. Because a downstream transition's velocity
    feeds an upstream transition's cost, a link committed in one sweep is **swapped**
    (re-assigned to a different successor) in a later sweep when whole-sequence
    evidence contradicts it — the retroactive error-correction the greedy LAP lacks.

    **Distinct from the rejected single-shot flow (SOT-2830/2840).** That min-cost
    flow solved each frame-pair assignment *independently* (its cost was separable
    per transition, so the "window" never bit — see :attr:`global_window`). This axis
    is genuinely coupled across transitions by the second-order motion term and
    performs re-assignment (swaps), which a single independent LAP-with-birth/death
    cannot. It is also distinct from the SOT-2864 global smoothed *forward-only*
    motion field and the SOT-2911 spatial-affine field: those re-rank a single
    forward LAP, they do not re-optimise the whole sequence with backward coupling.

    Pure numpy/scipy, CPU, offline, deterministic. Linking-only (detection is
    champion-invariant). Emits only ``t -> t+1`` metric-valid one-to-one edges; the
    short-track prune and division overlay run afterwards exactly as on the other
    global paths. Incompatible with in-linker division / gap-closing (ignored on this
    path). Reuses :attr:`viterbi_motion_gain` / :attr:`viterbi_curvature_weight` /
    :attr:`viterbi_theta` / :attr:`viterbi_max_sweeps`."""

    viterbi_motion_gain: float = 1.0
    """Velocity-extrapolation gain for the Viterbi motion trellis (SOT-2918).

    Only used when :attr:`viterbi_link` is set. A source's predicted next position is
    ``pos + viterbi_motion_gain * v_in`` (``v_in`` its current incoming displacement),
    and a destination's predicted origin is ``pos - viterbi_motion_gain * v_out``.
    ``0.0`` collapses both predictions to the raw positions, so the linker reduces to
    a whole-sequence distance LAP (no motion coupling)."""

    viterbi_curvature_weight: float = 1.0
    """Weight on the motion-incoherence (curvature) penalty (SOT-2918).

    Only used when :attr:`viterbi_link` is set. The assignment cost of a pair is
    ``dist + viterbi_curvature_weight * (fwd_residual + bwd_residual)`` where the two
    residuals are the scaled distances from the source's forward-predicted position
    and the destination's backward-predicted origin to the actual endpoints. ``0.0``
    drops the coupling (pure distance LAP over the whole sequence); larger values
    prefer trajectories whose velocity varies smoothly across the whole track. The
    ``<= max_distance`` feasibility gate stays on the **raw** scaled distance, so the
    motion term only re-ranks the champion's feasible set (never a new long edge)."""

    viterbi_theta: float = float("inf")
    """Birth+death link-acceptance threshold for the Viterbi linker (SOT-2918).

    Only used when :attr:`viterbi_link` is set. A ``t -> t+1`` pair is accepted only
    when its effective (distance + curvature) cost is ``< viterbi_theta`` — the
    Magnusson appearance/disappearance arc cost, letting the global linker *refuse* a
    marginal link and start/end a track instead. ``inf`` (default) accepts every
    distance-feasible pair (birth/death never cheaper), so acceptance is governed by
    the ``<= max_distance`` gate alone."""

    viterbi_max_sweeps: int = 8
    """Maximum whole-sequence re-optimisation sweeps for the Viterbi linker (SOT-2918).

    Only used when :attr:`viterbi_link` is set. Each sweep re-solves every transition
    against the previous sweep's velocities (Jacobi iteration); the loop stops early
    once the assignment reaches a fixed point (no pair changes), so this only caps a
    non-converging sequence. ``1`` performs a single motion-coupled pass (no swap)."""

    min_track_length: int = 1
    """Drop weakly-connected track fragments with fewer than this many nodes
    (SOT-2369, ported from the reference tracker's ``FILTER_SHORT_TRACKS``).

    A real cell persists across many frames, so a detection that fails to link
    into a multi-frame track is almost always noise. After linking, each track is
    a weakly-connected component; components with ``< min_track_length`` nodes are
    removed entirely (their nodes and edges). This is decoupled from detection
    thresholding: it prunes spurious detections *by tracking topology* rather than
    by intensity. Because a light-sheet volume is annotated sparsely, an
    over-predicting champion pays a node-count penalty
    ``J_adj = J·(1 - 0.1·(N_pred - N_true)/N_true)``; dropping isolated singletons
    (``min_track_length=2``) removes nodes that carry **no** matched edge, so edge
    TP/FP/FN are untouched while ``N_pred`` falls — a strict adjusted-Jaccard gain
    whenever the pipeline over-predicts. ``1`` keeps every node (champion
    behaviour, byte-for-byte)."""


def _suspicious_edge_review(
    graph: TrackingGraph,
    scale: np.ndarray,
    turn_cos: float,
    jump_ratio: float,
    jump_floor: float,
) -> TrackingGraph:
    """Cut post-hoc "suspicious" links (SOT-2895; dalloliogm event-review port).

    An interior link ``u -> v`` is removed when, given ``u``'s single incoming edge
    ``p -> u``, the outgoing displacement both **reverses** the incoming direction
    (turn cosine ``< turn_cos``) and **jumps** in length
    (``|d2| > jump_ratio * max(|d1|, jump_floor)``), with distances in scaled
    microns. This is the teleport-to-decoy signature of a false-positive link; the
    real trajectory is contradicted by its own finished motion. Only single-parent,
    single-child interior nodes are reviewed (``p`` and ``v`` unique), so a genuine
    division vertex (out-degree ``>= 2``) is never touched. Returns a new graph with
    the same nodes and the surviving edges; a no-op returns an equivalent graph.
    """
    dropped: set[tuple[int, int]] = set()
    for u in graph.node_ids():
        preds = graph.predecessors(u)
        succs = graph.successors(u)
        if len(preds) != 1 or len(succs) != 1:
            continue  # only review the interior of a simple chain
        p = preds[0]
        v = succs[0]
        d1 = (graph.position(u) - graph.position(p)) * scale
        d2 = (graph.position(v) - graph.position(u)) * scale
        s1 = float(np.sqrt((d1 * d1).sum()))
        s2 = float(np.sqrt((d2 * d2).sum()))
        if s1 <= 0.0 or s2 <= 0.0:
            continue
        cos = float((d1 * d2).sum()) / (s1 * s2)
        is_reversal = cos < turn_cos
        is_jump = s2 > jump_ratio * max(s1, jump_floor)
        if is_reversal and is_jump:
            dropped.add((u, v))
    if not dropped:
        return graph
    kept_edges = [e for e in graph.edges if e not in dropped]
    return TrackingGraph.from_lists(dict(graph.coords), kept_edges)


def _prune_short_tracks(graph: TrackingGraph, min_nodes: int) -> TrackingGraph:
    """Drop weakly-connected components smaller than ``min_nodes`` nodes."""
    if min_nodes <= 1:
        return graph
    parent: dict[int, int] = {n: n for n in graph.node_ids()}

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:  # path compression
            parent[a], a = root, parent[a]
        return root

    for src, dst in graph.edges:
        parent[find(src)] = find(dst)

    sizes: dict[int, int] = {}
    for n in graph.node_ids():
        r = find(n)
        sizes[r] = sizes.get(r, 0) + 1
    keep = [n for n in graph.node_ids() if sizes[find(n)] >= min_nodes]
    return graph.subgraph(keep)


def _gap_close(
    graph: TrackingGraph,
    scale: np.ndarray,
    max_frame_gap: int,
    gap_distance: float,
) -> TrackingGraph:
    """Bridge track-fragment terminals across missing frames (SOT-2763).

    The classical Jaqaman/TrackMate second LAP step. A **tail** (node with no
    successor) at timepoint ``t`` is joined to a later fragment **head** (node
    with no predecessor) at ``t + g`` for a frame gap ``2 <= g <= max_frame_gap``
    whose scaled tail->head distance is ``<= gap_distance``. The bridge edges are
    chosen by an **optimal min-cost assignment** (each tail bridges to at most one
    head and vice versa). All bridges point strictly forward in time, so the graph
    stays acyclic and no fragment is bridged to itself.

    The assignment is solved per connected component of the feasible-pair
    bipartite graph. Because ``gap_distance`` and ``max_frame_gap`` are small, the
    feasible pairs are spatially/temporally local and decompose into many tiny
    components; a block-diagonal LAP equals the global LAP but each block is small,
    so this stays tractable on dense volumes (tens of thousands of nodes). Mutates
    *graph* in place (adds edges) and returns it.
    """
    if max_frame_gap <= 1:
        return graph
    tails = [n for n in graph.node_ids() if graph.out_degree(n) == 0]
    heads = [n for n in graph.node_ids() if graph.in_degree(n) == 0]
    if not tails or not heads:
        return graph

    heads_by_t: dict[int, list[int]] = {}
    for h in heads:
        heads_by_t.setdefault(graph.t(h), []).append(h)

    # Feasible (tail, head, cost) candidates: cost is the scaled tail->head
    # distance; a candidate is admissible only within the frame-gap and distance
    # gates. Vectorised over each (tail-time, gap) block of heads.
    cand: list[tuple[int, int, float]] = []
    for tail in tails:
        t0 = graph.t(tail)
        p0 = graph.position(tail) * scale
        for g in range(2, max_frame_gap + 1):
            hs = heads_by_t.get(t0 + g)
            if not hs:
                continue
            hpos = np.array([graph.position(h) for h in hs], dtype=float) * scale
            dist = np.sqrt(((hpos - p0) ** 2).sum(axis=1))
            for h, d in zip(hs, dist):
                if h != tail and d <= gap_distance:
                    cand.append((tail, h, float(d)))
    if not cand:
        return graph

    # Union-find over the feasible-pair bipartite graph (tail/head namespaced so
    # an isolated node — both a tail and a head — never collides with itself).
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for tail, h, _cost in cand:
        parent[find(("t", tail))] = find(("h", h))

    comps: dict[tuple[str, int], list[tuple[int, int, float]]] = {}
    for tail, h, cost in cand:
        comps.setdefault(find(("t", tail)), []).append((tail, h, cost))

    new_edges: list[tuple[int, int]] = []
    big = gap_distance * 1000.0 + 1.0
    for pairs in comps.values():
        comp_tails = sorted({p[0] for p in pairs})
        comp_heads = sorted({p[1] for p in pairs})
        ti = {n: i for i, n in enumerate(comp_tails)}
        hi = {n: i for i, n in enumerate(comp_heads)}
        cost = np.full((len(comp_tails), len(comp_heads)), big, dtype=float)
        for tail, h, c in pairs:
            r, col = ti[tail], hi[h]
            if c < cost[r, col]:
                cost[r, col] = c
        rows, cols = linear_sum_assignment(cost)
        for r, col in zip(rows, cols):
            if cost[r, col] < big:
                new_edges.append((comp_tails[r], comp_heads[col]))

    for src, dst in new_edges:
        graph.add_edge(src, dst)
    return graph


def _component_sizes(graph: TrackingGraph) -> dict[int, int]:
    """Weakly-connected component size for every node id (union-find over edges)."""
    parent: dict[int, int] = {n: n for n in graph.node_ids()}

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:  # path compression
            parent[a], a = root, parent[a]
        return root

    for src, dst in graph.edges:
        parent[find(src)] = find(dst)
    sizes: dict[int, int] = {}
    for n in graph.node_ids():
        sizes[find(n)] = sizes.get(find(n), 0) + 1
    return {n: sizes[find(n)] for n in graph.node_ids()}


def _gap_recover(
    graph: TrackingGraph,
    scale: np.ndarray,
    max_gap: int,
    distance: float,
    min_frag: int,
) -> TrackingGraph:
    """Node-interpolation gap recovery across missing detection frames (SOT-2849).

    Distinct from :func:`_gap_close` (SOT-2763), which added a single
    *non-consecutive* bridge edge that the competition edge metric drops. Here,
    when a track-fragment tail at ``t`` is reconnected to a later fragment head at
    ``t + g`` (``2 <= g <= max_gap``) within ``distance`` microns, we **insert
    interpolated detection nodes** at each missing frame ``t+1 .. t+g-1`` (linear
    interpolation of the voxel centroid between tail and head) and wire them into a
    chain of *consecutive* edges ``tail -> interp -> ... -> head``. Those
    consecutive edges ARE scored, and an interpolated node that lands within the
    ``<= 7 µm`` per-timepoint match radius of the true (missed-detection) GT node
    recovers a real FN edge.

    ``min_frag`` gates eligibility on the weakly-connected fragment size at each
    terminal (both the tail's and the head's fragment must have ``>= min_frag``
    nodes), to avoid resurrecting the short noise fragments the champion's
    ``min_track_length`` prune is designed to remove (the SOT-2763 failure mode).
    Bridges are chosen by an optimal min-cost assignment (each tail bridges to at
    most one head and vice versa). Mutates *graph* in place and returns it.
    """
    if max_gap <= 1:
        return graph
    tails = [n for n in graph.node_ids() if graph.out_degree(n) == 0]
    heads = [n for n in graph.node_ids() if graph.in_degree(n) == 0]
    if not tails or not heads:
        return graph

    if min_frag > 1:
        comp_size = _component_sizes(graph)
        tails = [n for n in tails if comp_size.get(n, 1) >= min_frag]
        heads = [n for n in heads if comp_size.get(n, 1) >= min_frag]
        if not tails or not heads:
            return graph

    heads_by_t: dict[int, list[int]] = {}
    for h in heads:
        heads_by_t.setdefault(graph.t(h), []).append(h)

    # Feasible (tail, head, gap, cost) candidates within the frame-gap and distance
    # gates; cost is the scaled tail->head distance.
    cand: list[tuple[int, int, int, float]] = []
    for tail in tails:
        t0 = graph.t(tail)
        p0 = graph.position(tail) * scale
        for g in range(2, max_gap + 1):
            hs = heads_by_t.get(t0 + g)
            if not hs:
                continue
            hpos = np.array([graph.position(h) for h in hs], dtype=float) * scale
            dist = np.sqrt(((hpos - p0) ** 2).sum(axis=1))
            for h, d in zip(hs, dist):
                if h != tail and d <= distance:
                    cand.append((tail, h, g, float(d)))
    if not cand:
        return graph

    # Union-find over the feasible-pair bipartite graph (tail/head namespaced), then
    # solve the min-cost assignment per connected component (block-diagonal == global
    # LAP; each block is small because the gates are spatially/temporally local).
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for tail, h, _g, _c in cand:
        parent[find(("t", tail))] = find(("h", h))

    comps: dict[tuple[str, int], list[tuple[int, int, int, float]]] = {}
    for tail, h, g, c in cand:
        comps.setdefault(find(("t", tail)), []).append((tail, h, g, c))

    big = distance * 1000.0 + 1.0
    chosen: list[tuple[int, int, int]] = []  # (tail, head, gap)
    for pairs in comps.values():
        comp_tails = sorted({p[0] for p in pairs})
        comp_heads = sorted({p[1] for p in pairs})
        ti = {n: i for i, n in enumerate(comp_tails)}
        hi = {n: i for i, n in enumerate(comp_heads)}
        cost = np.full((len(comp_tails), len(comp_heads)), big, dtype=float)
        gap_of: dict[tuple[int, int], int] = {}
        for tail, h, g, c in pairs:
            r, col = ti[tail], hi[h]
            if c < cost[r, col]:
                cost[r, col] = c
                gap_of[(r, col)] = g
        rows, cols = linear_sum_assignment(cost)
        for r, col in zip(rows, cols):
            if cost[r, col] < big:
                chosen.append((comp_tails[r], comp_heads[col], gap_of[(r, col)]))
    if not chosen:
        return graph

    # Insert interpolated nodes (fresh ids) and wire the consecutive chain.
    next_id = max(graph.node_ids()) + 1
    for tail, head, g in chosen:
        t0 = graph.t(tail)
        p_tail = graph.position(tail)  # voxel coords
        p_head = graph.position(head)
        prev = tail
        for step in range(1, g):
            frac = step / g
            pos = p_tail + frac * (p_head - p_tail)
            graph.add_node(
                next_id, float(t0 + step), float(pos[0]), float(pos[1]), float(pos[2])
            )
            graph.add_edge(prev, next_id)
            prev = next_id
            next_id += 1
        graph.add_edge(prev, head)
    return graph


def _appearance_cost(
    src_desc: np.ndarray, dst_desc: np.ndarray, weight: float
) -> np.ndarray:
    """``weight * (1 - similarity)`` appearance penalty matrix (SOT-2829).

    ``similarity`` is the cosine of the standardised descriptor vectors mapped to
    ``[0, 1]`` via ``(1 + cos) / 2``, so the penalty lies in ``[0, weight]`` — a
    look-alike successor pays ~0, a dissimilar one pays up to ``weight`` microns of
    extra cost. A zero-norm descriptor (a perfectly flat patch) yields similarity
    ``0.5`` (neutral), never a divide-by-zero.
    """
    sn = np.linalg.norm(src_desc, axis=1)
    dn = np.linalg.norm(dst_desc, axis=1)
    denom = sn[:, None] * dn[None, :]
    dots = src_desc @ dst_desc.T
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(denom > 1e-12, dots / denom, 0.0)
    cos = np.clip(cos, -1.0, 1.0)
    similarity = 0.5 * (1.0 + cos)
    return weight * (1.0 - similarity)


def _assign(
    src: np.ndarray, dst: np.ndarray, scale: np.ndarray, max_distance: float,
    src_pred: np.ndarray | None = None, disp_weight: float = 0.0,
    gate_on_prediction: bool = False,
    src_desc: np.ndarray | None = None, dst_desc: np.ndarray | None = None,
    appearance_weight: float = 0.0,
    edge_model=None,
    edge_gate_expand: bool = False,
    edge_gate_admit_prob: float = 0.5,
    edge_gate_expand_ratio: float = 1.5,
    xattn_model=None,
    dst_pred_bwd: np.ndarray | None = None,
    consistency_gate: bool = False,
    consistency_tol: float = float("inf"),
    consistency_weight: float = 0.0,
) -> list[tuple[int, int]]:
    """Optimal one-to-one assignment of ``src`` rows to ``dst`` rows within range.

    Returns a list of ``(src_index, dst_index)`` pairs whose scaled distance is
    ``<= max_distance``. Costs above the threshold are masked to a large finite
    value so the assignment never prefers an out-of-range pair over an in-range
    one, then filtered out after solving.

    When ``src_pred`` is given (constant-velocity predicted source positions),
    the assignment *cost* is the scaled distance from the **predicted** source to
    the destination plus ``disp_weight`` times the raw (unpredicted) scaled
    distance; the feasibility gate uses the predicted distance if
    ``gate_on_prediction`` else the actual distance.

    When ``appearance_weight > 0`` and per-detection descriptors are supplied, an
    ``appearance_weight * (1 - similarity)`` term (:func:`_appearance_cost`) is
    **added to the cost only** (SOT-2829); the feasibility gate stays on the scaled
    distance, so appearance re-ranks within the champion's feasible set but never
    admits an out-of-range edge.

    When ``edge_model`` is a :class:`biohub_tracking.edge_linker.LearnedEdgeCost`
    with descriptors supplied, its GT-learned ``weight * (1 - p_edge)`` penalty
    (SOT-2841) is likewise **added to the cost only**; the gate stays on the scaled
    distance, so it re-ranks the feasible set without admitting an out-of-range edge.
    """
    if len(src) == 0 or len(dst) == 0:
        return []
    diff = (src[:, None, :] - dst[None, :, :]) * scale
    dist = np.sqrt((diff**2).sum(axis=2))
    dist_pred: np.ndarray | None = None
    if src_pred is None:
        gate = dist
        cost_base = dist
    else:
        diff_pred = (src_pred[:, None, :] - dst[None, :, :]) * scale
        dist_pred = np.sqrt((diff_pred**2).sum(axis=2))
        gate = dist_pred if gate_on_prediction else dist
        cost_base = dist_pred + disp_weight * dist
    has_desc = src_desc is not None and dst_desc is not None
    if appearance_weight > 0.0 and has_desc:
        cost_base = cost_base + _appearance_cost(src_desc, dst_desc, appearance_weight)
    edge_active = edge_model is not None and edge_model.weight != 0.0 and has_desc
    if edge_active:
        # dist_pred (when a motion prediction is active) threads the motion-residual
        # feature into the joint edge vector, matching how the model was trained.
        cost_base = cost_base + edge_model.penalty(
            dist, src_desc, dst_desc, max_distance, dist_pred=dist_pred
        )
    # Learned CROSS-ATTENTION edge cost (SOT-2994): a sibling re-rank term whose
    # p_edge is contextualised by attention over the candidate set. Added to the cost
    # only (feasibility gate unchanged) exactly like the SOT-2841 edge_model, so it is
    # metric-valid and byte-for-byte inert at weight 0 / no model / no descriptors.
    xattn_active = xattn_model is not None and xattn_model.weight != 0.0 and has_desc
    if xattn_active:
        cost_base = cost_base + xattn_model.penalty(
            dist, src_desc, dst_desc, max_distance, dist_pred=dist_pred
        )

    # Base feasibility: the champion's raw (or, under gate_on_prediction, predicted)
    # distance gate.
    feasible = gate <= max_distance

    # SOT-2883 Ultrack bidirectional motion-consistency gate. Requires the SOT-2864
    # forward motion prediction (dist_pred = r_f, ``src_pred`` -> dst residual) and a
    # backward-field prediction of dst (``dst_pred_bwd`` -> src residual r_b). A link
    # is bidirectionally consistent only when BOTH direction residuals are small; the
    # soft term discounts consistent pairs in the cost, the hard tol drops
    # inconsistent ones from the feasible set (a pure restriction — never admits a new
    # pair, so the max_distance cap is preserved). Off (or missing dst_pred_bwd) =>
    # champion / SOT-2864 byte-for-byte.
    if consistency_gate and dist_pred is not None and dst_pred_bwd is not None:
        diff_bwd = (dst_pred_bwd[None, :, :] - src[:, None, :]) * scale
        r_b = np.sqrt((diff_bwd**2).sum(axis=2))
        r_f = dist_pred
        if consistency_weight > 0.0:
            cost_base = cost_base + consistency_weight * 0.5 * (r_f + r_b)
        if np.isfinite(consistency_tol):
            feasible = feasible & (r_f <= consistency_tol) & (r_b <= consistency_tol)

    # SOT-2870 learned gate EXPANSION (FN-edge recovery): additionally admit a
    # raw-far pair only when it is motion-corrected in-range, within the raw-distance
    # ratio cap, AND the learned classifier scores it a real edge. Requires an active
    # edge model and a motion prediction (dist_pred); off => champion byte-for-byte.
    if edge_gate_expand and edge_active and dist_pred is not None:
        p_edge = edge_model.probability_planes(
            dist, src_desc, dst_desc, max_distance, dist_pred=dist_pred
        )
        expand = (
            (~feasible)
            & (dist_pred <= max_distance)
            & (dist <= max_distance * edge_gate_expand_ratio)
            & (p_edge >= edge_gate_admit_prob)
        )
        feasible = feasible | expand

    big = max_distance * 1000.0 + float(cost_base.max()) + 1.0
    cost = np.where(feasible, cost_base, big)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if bool(feasible[r, c])]


def _two_pass_assign(
    src: np.ndarray,
    dst: np.ndarray,
    scale: np.ndarray,
    tight_gate: float,
    full_gate: float,
) -> list[tuple[int, int]]:
    """Classical-baseline two-pass Hungarian assignment (SOT-2899).

    Pass 1 solves the optimal one-to-one assignment within ``tight_gate`` microns
    (the champion :attr:`~LinkParams.max_distance`), making only high-confidence
    links. Pass 2 re-solves an optimal assignment on the **still-unmatched** sources
    and destinations only, out to ``full_gate`` microns, recovering a fast cell whose
    real successor sits just beyond the tight gate without perturbing any Pass-1 link.

    Returns ``(src_index, dst_index)`` pairs in the original array indexing. With
    ``full_gate <= tight_gate`` (or no leftovers) Pass 2 adds nothing, so the result
    equals the single-pass :func:`_assign` at ``tight_gate`` — byte-for-byte champion.
    """
    pairs = _assign(src, dst, scale, tight_gate)
    if full_gate <= tight_gate or len(src) == 0 or len(dst) == 0:
        return pairs
    matched_src = {i for i, _ in pairs}
    matched_dst = {j for _, j in pairs}
    free_src = [i for i in range(len(src)) if i not in matched_src]
    free_dst = [j for j in range(len(dst)) if j not in matched_dst]
    if not free_src or not free_dst:
        return pairs
    sub = _assign(src[free_src], dst[free_dst], scale, full_gate)
    for a, b in sub:
        pairs.append((free_src[a], free_dst[b]))
    return pairs


def _cycle_consistency_filter(
    pairs: list[tuple[int, int]],
    src: np.ndarray,
    dst: np.ndarray,
    scale: np.ndarray,
    margin: float,
) -> list[tuple[int, int]]:
    """Keep only mutually-consistent (mutual nearest-neighbour) links (SOT-2910).

    Given the primary ``t -> t+1`` assignment ``pairs`` (each ``(src_idx, dst_idx)``),
    a link ``i -> j`` survives only when it is a **mutual nearest neighbour** in scaled
    microns: ``j`` is ``i``'s forward-nearest destination AND ``i`` is ``j``'s
    backward-nearest source. This is the bidirectional cycle-consistency principle
    (NeighborTrack / DistNet2D): a globally-assigned but non-mutual link is a contested
    steal the two directions disagree on, so pruning it sheds an FP edge (and, after
    :attr:`~LinkParams.min_track_length` pruning, frees the matched GT node).

    ``margin > 0`` additionally drops a mutual-best link that is *contested* — its
    runner-up competitor lies within ``margin`` in either direction. The filter only
    ever **removes** links (never adds one and never widens the feasibility gate), so it
    is a strict restriction of the champion assignment; an empty ``pairs`` or a
    single-sided frame returns it unchanged. Deterministic (``np.argmin`` ties break to
    the first index, matching the assignment's own ordering).
    """
    if not pairs or len(src) == 0 or len(dst) == 0:
        return pairs
    diff = (src[:, None, :] - dst[None, :, :]) * scale
    dist = np.sqrt((diff**2).sum(axis=2))  # (Ns, Nd) scaled distances
    fwd_nn = np.argmin(dist, axis=1)  # each source's nearest destination
    bwd_nn = np.argmin(dist, axis=0)  # each destination's nearest source
    row2 = np.partition(dist, 1, axis=1)[:, 1] if margin > 0.0 and dist.shape[1] > 1 else None
    col2 = np.partition(dist, 1, axis=0)[1, :] if margin > 0.0 and dist.shape[0] > 1 else None
    kept: list[tuple[int, int]] = []
    for i, j in pairs:
        if int(fwd_nn[i]) != j or int(bwd_nn[j]) != i:
            continue  # non-mutual: the two directions disagree -> FP-prone steal
        if margin > 0.0:
            d_ij = dist[i, j]
            r2 = row2[i] if row2 is not None else float("inf")
            c2 = col2[j] if col2 is not None else float("inf")
            if (r2 - d_ij) < margin or (c2 - d_ij) < margin:
                continue  # mutual best but contested within the ambiguity margin
        kept.append((i, j))
    return kept


def _motion_field_predict(
    src: np.ndarray,
    dst: np.ndarray,
    scale: np.ndarray,
    max_distance: float,
    smooth_sigma: float,
    gain: float,
) -> np.ndarray:
    """Predicted ``src`` positions from a global, spatially-smoothed motion field.

    ARGUS-style (arXiv:2607.08297) motion-model prediction, numpy/scipy-only (the
    portable proxy for a Farneback dense optical flow when ``cv2`` is unavailable):

    1. Run a provisional optimal assignment ``src -> dst`` within ``max_distance``
       (:func:`_assign`) to obtain *anchor* displacements ``v_k = dst[j] - src[i]``
       (voxel space) located at each matched source ``src[i]``.
    2. For every source detection ``i`` — including ones the provisional pass left
       unmatched — set its predicted displacement to the anchor displacements
       weighted by a Gaussian of the *scaled* distance to each anchor source,
       ``w = exp(-0.5 * (d / smooth_sigma)**2)``. This diffuses the sparse anchor
       flow into a dense, locally-consistent field, so a first-appearance cell with
       no track history still inherits its neighbourhood's motion.
    3. Return ``src + gain * predicted_displacement`` (voxel space).

    Deterministic and pure numpy/scipy. With no anchors (or ``gain == 0``) the
    predicted positions equal ``src`` (the assignment then reduces to the champion
    nearest-neighbour path).
    """
    pred = np.asarray(src, dtype=float).copy()
    if gain == 0.0 or len(src) == 0 or len(dst) == 0:
        return pred
    anchors = _assign(src, dst, scale, max_distance)
    if not anchors:
        return pred
    anchor_src = np.array([src[i] for i, _ in anchors], dtype=float)
    anchor_disp = np.array([dst[j] - src[i] for i, j in anchors], dtype=float)
    if smooth_sigma <= 0.0:
        # No spatial smoothing: each anchored source uses its own displacement,
        # every other source keeps zero (nearest-anchor fallback below is skipped).
        for k, (i, _) in enumerate(anchors):
            pred[i] = src[i] + gain * anchor_disp[k]
        return pred
    # Gaussian-weighted average of anchor displacements over scaled distance.
    diff = (src[:, None, :] - anchor_src[None, :, :]) * scale  # (N, K, 3)
    d2 = (diff**2).sum(axis=2)  # (N, K) squared scaled distance
    w = np.exp(-0.5 * d2 / (smooth_sigma * smooth_sigma))  # (N, K)
    wsum = w.sum(axis=1, keepdims=True)  # (N, 1)
    # Rows with negligible total weight (far from every anchor) keep zero motion.
    safe = wsum[:, 0] > 1e-12
    field = np.zeros_like(pred)
    field[safe] = (w[safe][:, :, None] * anchor_disp[None, :, :]).sum(axis=1) / wsum[safe]
    pred = pred + gain * field
    return pred


def _local_affine_predict(
    src: np.ndarray,
    dst: np.ndarray,
    scale: np.ndarray,
    max_distance: float,
    k: int,
    gain: float,
    ridge: float,
) -> np.ndarray:
    """Predicted ``src`` positions from a per-cell **local affine** motion field.

    Collective-motion affine analysis (PLOS Comput. Biol. pcbi.1008407 / PMC9287486)
    and Amat super-voxel optical flow (Bioinformatics 29(3):373), numpy/scipy-only:

    1. Run a provisional optimal assignment ``src -> dst`` within ``max_distance``
       (:func:`_assign`) to obtain *anchor* displacements ``v_k = dst[j] - src[i]``
       (voxel space) at each matched source ``src[i]`` — the same provisional flow the
       SOT-2864 global field uses.
    2. For every source detection ``i`` (matched or not), take its ``k`` nearest
       anchors by *scaled* distance and fit a local affine map ``disp ≈ A·p + b`` (p =
       scaled position) by ridge least squares, then predict ``disp_i = A·p_i + b``.
       With ``< 4`` usable anchors the fit degenerates to the local mean displacement
       (a pure translation); with no anchor in reach the cell keeps zero motion.
    3. Return ``src + gain * predicted_displacement`` (voxel space).

    Unlike the global Gaussian-smoothed field (:func:`_motion_field_predict`, a locally
    *constant* translation), the affine term ``A`` represents the first-order velocity
    gradient (shear / divergence) of neighbouring tissue patches. Deterministic and
    pure numpy/scipy. With no anchors (or ``gain == 0``) the predicted positions equal
    ``src`` (the assignment reduces to the champion nearest-neighbour path).
    """
    pred = np.asarray(src, dtype=float).copy()
    if gain == 0.0 or len(src) == 0 or len(dst) == 0 or k <= 0:
        return pred
    anchors = _assign(src, dst, scale, max_distance)
    if not anchors:
        return pred
    anchor_src = np.array([src[i] for i, _ in anchors], dtype=float)  # (K, 3) voxel
    anchor_disp = np.array(
        [dst[j] - src[i] for i, j in anchors], dtype=float
    )  # (K, 3) voxel displacement
    anchor_pos = anchor_src * scale  # (K, 3) scaled position (fit basis)
    src_pos = src * scale  # (N, 3) scaled
    kk = min(int(k), len(anchors))
    field = np.zeros_like(pred)
    for i in range(len(src)):
        d2 = ((anchor_pos - src_pos[i][None, :]) ** 2).sum(axis=1)  # (K,) scaled dist^2
        nn = np.argsort(d2)[:kk]
        p = anchor_pos[nn]  # (kk, 3) scaled positions
        v = anchor_disp[nn]  # (kk, 3) voxel displacements
        if len(nn) < 4:
            # Too few neighbours for a stable affine: local mean translation.
            field[i] = v.mean(axis=0)
            continue
        # Ridge-regularised affine fit disp ≈ A·p + b via the augmented design
        # matrix [p | 1]; the bias column is left un-penalised so the fit can still
        # reproduce a pure translation exactly (only the linear A block is shrunk).
        design = np.concatenate([p, np.ones((len(nn), 1))], axis=1)  # (kk, 4)
        gram = design.T @ design  # (4, 4)
        reg = np.eye(4) * float(ridge)
        reg[3, 3] = 0.0  # do not penalise the bias/translation term
        try:
            coef = np.linalg.solve(gram + reg, design.T @ v)  # (4, 3)
        except np.linalg.LinAlgError:
            field[i] = v.mean(axis=0)
            continue
        q = np.concatenate([src_pos[i], [1.0]])  # (4,) this cell's scaled position + 1
        field[i] = q @ coef  # (3,) predicted voxel displacement
    pred = pred + gain * field
    return pred


def _global_assign(
    src: np.ndarray, dst: np.ndarray, scale: np.ndarray, max_distance: float,
    theta: float,
) -> list[tuple[int, int]]:
    """One transition of the global min-cost-flow assignment (SOT-2830).

    Solves the assignment-with-**birth/death-outliers** for a single ``t -> t+1``
    transition: each ``src`` row may link to at most one ``dst`` row (and vice
    versa) at cost = scaled distance, or stay unlinked — the source *dies*
    (:attr:`LinkParams.death_cost`) and the destination is *born*
    (:attr:`LinkParams.birth_cost`). Only ``theta = death_cost + birth_cost``
    enters, as the link-acceptance threshold: a pair is linked only when it is
    cheaper than paying both outlier costs, i.e. its scaled distance is
    ``< theta`` (and ``<= max_distance``).

    This is the exact single-transition min-cost flow (the block flow decouples per
    transition for a pure-distance cost — see :attr:`LinkParams.global_window`), and
    it stays a rectangular ``len(src) x len(dst)`` LAP (same size/cost as the
    per-frame :func:`_assign`, no dummy-padded square blow-up).

    ``theta = inf`` reproduces :func:`_assign` **exactly** (every feasible pair is
    cheaper than an infinite outlier, so the champion matching is recovered),
    keeping the global path a strict generalisation of the per-frame champion.
    """
    if not np.isfinite(theta):
        # Infinite outlier cost => every feasible link is worthwhile; this is the
        # champion's optimal in-range matching, byte-for-byte.
        return _assign(src, dst, scale, max_distance)
    if len(src) == 0 or len(dst) == 0:
        return []
    diff = (src[:, None, :] - dst[None, :, :]) * scale
    dist = np.sqrt((diff**2).sum(axis=2))
    feasible = dist <= max_distance
    # Profit of a link vs. leaving both endpoints as death+birth is
    # ``theta - dist`` (> 0 only when dist < theta). Maximise total profit ==
    # minimise total ``min(dist - theta, 0)``; a forced 0-profit filler pair
    # (dist >= theta or infeasible) contributes nothing and is dropped below, so
    # the rectangular LAP still yields the optimal outlier assignment.
    cost = np.where(feasible, np.minimum(dist - theta, 0.0), 0.0)
    rows, cols = linear_sum_assignment(cost)
    return [
        (int(r), int(c))
        for r, c in zip(rows, cols)
        if feasible[r, c] and dist[r, c] < theta
    ]


def _global_link(
    graph: TrackingGraph,
    detections: dict[int, np.ndarray],
    ids_by_t: dict[int, list[int]],
    scale_arr: np.ndarray,
    params: LinkParams,
) -> None:
    """Global short-window min-cost-flow linking (SOT-2830), in place.

    Adds one-to-one ``t -> t+1`` edges chosen by the birth/death outlier
    assignment (:func:`_global_assign`) for every consecutive-frame transition, in
    time-then-source order (so edge insertion order matches the per-frame champion
    when ``theta == inf``). Emits **only consecutive-frame edges** — no bridge/gap
    edge (metric-valid) — and does not attach division daughters, so the global
    path is a clean, metric-continuous generalisation of the per-frame linker whose
    sole added lever is the birth/death threshold ``theta``.
    """
    theta = params.birth_cost + params.death_cost
    times = sorted(detections)
    for t_a, t_b in zip(times, times[1:]):
        if t_b != t_a + 1:
            continue  # only link consecutive timepoints (no bridge edges)
        pairs = _global_assign(
            detections[t_a], detections[t_b], scale_arr, params.max_distance, theta
        )
        for i, j in pairs:
            graph.add_edge(ids_by_t[t_a][i], ids_by_t[t_b][j])


def _viterbi_assign(
    src: np.ndarray,
    dst: np.ndarray,
    scale: np.ndarray,
    max_distance: float,
    theta: float,
    v_in: np.ndarray,
    v_out: np.ndarray,
    gain: float,
    curvature_weight: float,
) -> list[tuple[int, int]]:
    """One motion-coupled transition of the Viterbi global linker (SOT-2918).

    Solves the ``t -> t+1`` assignment on the Magnusson motion-coherence cost

    ``c(i, j) = ||src_i - dst_j|| + curvature_weight * (r_fwd + r_bwd)``

    where ``r_fwd = ||(src_i + gain*v_in_i) - dst_j||`` is the residual of the
    source's **forward** velocity prediction and ``r_bwd = ||src_i - (dst_j -
    gain*v_out_j)||`` the residual of the destination's **backward** velocity
    prediction (all in scaled microns). ``v_in`` / ``v_out`` are the per-detection
    incoming / outgoing displacements from the *previous* whole-sequence sweep
    (voxel units), which is what couples this transition to both its neighbours and
    lets a later sweep swap an earlier link. The feasibility gate stays on the
    **raw** scaled distance ``<= max_distance`` (motion only re-ranks, never admits a
    new long edge); a finite ``theta`` additionally refuses a pair whose effective
    cost is ``>= theta`` (birth/death acceptance). With ``v_in == v_out == 0`` (or
    ``gain == 0``) and ``theta == inf`` the cost is ``(1 + 2*curvature_weight)*dist``
    — the champion distance ranking — so the first sweep matches the champion
    feasible optimum before the coupling bites.
    """
    if len(src) == 0 or len(dst) == 0:
        return []
    diff = (src[:, None, :] - dst[None, :, :]) * scale
    dist = np.sqrt((diff**2).sum(axis=2))
    cost_base = dist
    if curvature_weight != 0.0 and gain != 0.0:
        pred_src = src + gain * v_in  # forward extrapolation of each source
        diff_f = (pred_src[:, None, :] - dst[None, :, :]) * scale
        r_fwd = np.sqrt((diff_f**2).sum(axis=2))
        pred_dst_origin = dst - gain * v_out  # where each dst is predicted to come from
        diff_b = (src[:, None, :] - pred_dst_origin[None, :, :]) * scale
        r_bwd = np.sqrt((diff_b**2).sum(axis=2))
        cost_base = dist + curvature_weight * (r_fwd + r_bwd)

    feasible = dist <= max_distance
    if np.isfinite(theta):
        feasible = feasible & (cost_base < theta)
    if not feasible.any():
        return []
    big = max_distance * 1000.0 + float(cost_base.max()) + 1.0
    cost = np.where(feasible, cost_base, big)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if bool(feasible[r, c])]


def _viterbi_link(
    graph: TrackingGraph,
    detections: dict[int, np.ndarray],
    ids_by_t: dict[int, list[int]],
    scale_arr: np.ndarray,
    params: LinkParams,
) -> None:
    """Whole-sequence Viterbi global track-linking with swaps (SOT-2918), in place.

    Ports Magnusson/Jaldén/Gilbert/Blau (IEEE TMI 2015). The greedy per-frame LAP
    commits each link once and propagates any early error; this instead re-solves the
    **whole sequence** by iterated re-optimisation. Each sweep re-runs every
    consecutive-frame :func:`_viterbi_assign` using the incoming/outgoing velocities
    from the *previous* sweep's assignment, so a downstream transition's velocity
    feeds an upstream transition's cost (and vice-versa). A link is therefore
    **swapped** — re-assigned to a different successor — in a later sweep when
    whole-sequence motion evidence contradicts it, the retroactive error-correction
    the greedy LAP lacks. The loop stops at the first fixed point (assignment
    unchanged) or after :attr:`LinkParams.viterbi_max_sweeps` sweeps.

    Emits only consecutive-frame one-to-one ``t -> t+1`` metric-valid edges (no
    bridge, no in-linker division), so the graph is a motion-coherent
    generalisation of the per-frame champion whose sole levers are the motion-
    coupling weight/gain and the birth/death threshold ``viterbi_theta``.
    """
    times = sorted(detections)
    transitions = [
        (t_a, t_b) for t_a, t_b in zip(times, times[1:]) if t_b == t_a + 1
    ]
    gain = params.viterbi_motion_gain
    cw = params.viterbi_curvature_weight
    theta = params.viterbi_theta
    max_sweeps = max(1, int(params.viterbi_max_sweeps))

    # Per-transition assignment (list of (i, j) index pairs), updated each sweep.
    pairs_by_trans: dict[int, list[tuple[int, int]]] = {
        k: [] for k in range(len(transitions))
    }

    for _sweep in range(max_sweeps):
        # Velocities from the *previous* sweep's assignment (Jacobi coupling). v_in
        # of a dst comes from the link that lands on it; v_out of a src from the link
        # leaving it. Displacements are in voxel units (scaled inside the cost).
        v_in_by_t: dict[int, np.ndarray] = {
            t: np.zeros_like(detections[t], dtype=float) for t in times
        }
        v_out_by_t: dict[int, np.ndarray] = {
            t: np.zeros_like(detections[t], dtype=float) for t in times
        }
        for k, (t_a, t_b) in enumerate(transitions):
            src, dst = detections[t_a], detections[t_b]
            for i, j in pairs_by_trans[k]:
                disp = dst[j] - src[i]
                v_in_by_t[t_b][j] = disp   # dst j arrived with this velocity
                v_out_by_t[t_a][i] = disp  # src i departs with this velocity

        new_pairs: dict[int, list[tuple[int, int]]] = {}
        for k, (t_a, t_b) in enumerate(transitions):
            new_pairs[k] = _viterbi_assign(
                detections[t_a], detections[t_b], scale_arr,
                params.max_distance, theta,
                v_in=v_in_by_t[t_a],    # src i's incoming vel (t_a-1 -> t_a link's disp)
                v_out=v_out_by_t[t_b],  # dst j's outgoing vel (t_b -> t_b+1 link's disp)
                gain=gain, curvature_weight=cw,
            )

        if new_pairs == pairs_by_trans:
            pairs_by_trans = new_pairs
            break  # fixed point reached: whole-sequence-consistent assignment
        pairs_by_trans = new_pairs

    for k, (t_a, t_b) in enumerate(transitions):
        for i, j in pairs_by_trans[k]:
            graph.add_edge(ids_by_t[t_a][i], ids_by_t[t_b][j])


def _window_assign(
    src: np.ndarray, dst: np.ndarray, scale: np.ndarray, max_distance: float,
    theta: float,
    src_pred: np.ndarray | None = None, disp_weight: float = 0.0,
    gate_on_prediction: bool = False,
    src_desc: np.ndarray | None = None, dst_desc: np.ndarray | None = None,
    appearance_weight: float = 0.0, edge_model=None,
) -> list[tuple[int, int]]:
    """One windowed-association transition: motion-predicted LAP with birth/death
    outliers (SOT-2871).

    Combines the SOT-2864 predicted-position cost (``src_pred``) with the SOT-2830
    birth/death link-acceptance threshold ``theta``: a pair is linked only when its
    **effective** cost (predicted-position distance ``+ disp_weight * raw`` plus any
    appearance/edge-model penalty) is ``< theta`` and its gate distance is
    ``<= max_distance``. With ``theta = inf``, ``src_pred = None`` and no penalty
    terms this is **byte-identical** to :func:`_assign` (the champion path), so the
    windowed path is a strict generalisation.
    """
    if len(src) == 0 or len(dst) == 0:
        return []
    diff = (src[:, None, :] - dst[None, :, :]) * scale
    dist = np.sqrt((diff**2).sum(axis=2))
    dist_pred: np.ndarray | None = None
    if src_pred is None:
        gate = dist
        cost_base = dist
    else:
        diff_pred = (src_pred[:, None, :] - dst[None, :, :]) * scale
        dist_pred = np.sqrt((diff_pred**2).sum(axis=2))
        gate = dist_pred if gate_on_prediction else dist
        cost_base = dist_pred + disp_weight * dist
    has_desc = src_desc is not None and dst_desc is not None
    if appearance_weight > 0.0 and has_desc:
        cost_base = cost_base + _appearance_cost(src_desc, dst_desc, appearance_weight)
    if edge_model is not None and edge_model.weight != 0.0 and has_desc:
        cost_base = cost_base + edge_model.penalty(
            dist, src_desc, dst_desc, max_distance, dist_pred=dist_pred
        )

    feasible = gate <= max_distance
    if np.isfinite(theta):
        # Birth/death acceptance: refuse a link costlier than the outlier sum, so a
        # source may die / a destination be born instead of taking a marginal edge.
        feasible = feasible & (cost_base < theta)

    big = max_distance * 1000.0 + float(cost_base.max()) + 1.0
    cost = np.where(feasible, cost_base, big)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if bool(feasible[r, c])]


def _parental_softmax_divide(
    graph: TrackingGraph,
    ids_a: list[int],
    ids_b: list[int],
    src: np.ndarray,
    dst: np.ndarray,
    scale: np.ndarray,
    params: LinkParams,
    pairs: list[tuple[int, int]],
) -> None:
    """Attach balanced second daughters under Trackastra's parental-softmax
    constraint (SOT-2871), in place.

    For each matched parent the association mass over its feasible children (those
    within :attr:`LinkParams.division_distance`) is a softmax of
    ``-scaled_distance / window_softmax_temp``, so the shares sum to ``<= 1`` (the
    parental-softmax "one parent" budget). A leftover ``t+1`` detection is attached
    as a parent's second daughter only when its softmax share is
    ``>= window_softmax_min_share`` — i.e. the parent's mass is genuinely split
    between two comparable daughters (a real mitosis), never leaked to a distant
    detection (a division FP). Out-degree stays capped at two.
    """
    matched_dst = {j for _, j in pairs}
    matched_src = {i for i, _ in pairs}
    if not matched_src:
        return
    parent_pos = src * scale
    temp = max(params.window_softmax_temp, 1e-6)
    # Feasible-child softmax shares per matched parent: over every child within
    # division_distance (scaled), mass = softmax(-dist / temp).
    dst_pos = dst * scale
    for j in range(len(dst)):
        if j in matched_dst:
            continue
        # Candidate parents (matched, out-degree < 2, within division_distance),
        # ranked by the share this leftover child would carry of the parent's mass.
        best_parent = -1
        best_share = 0.0
        for i in matched_src:
            if graph.out_degree(ids_a[i]) >= 2:
                continue
            d_ij = float(np.sqrt(((parent_pos[i] - dst_pos[j]) ** 2).sum()))
            if d_ij > params.division_distance:
                continue
            # Parent i's feasible children (within division_distance), incl. this one.
            d_children = np.sqrt(((dst_pos - parent_pos[i]) ** 2).sum(axis=1))
            feas = d_children <= params.division_distance
            if not feas.any():
                continue
            logits = -d_children[feas] / temp
            logits = logits - logits.max()
            w = np.exp(logits)
            w = w / w.sum()
            share = float(w[np.flatnonzero(feas) == j][0]) if feas[j] else 0.0
            if share > best_share:
                best_share = share
                best_parent = i
        if best_parent >= 0 and best_share >= params.window_softmax_min_share:
            graph.add_edge(ids_a[best_parent], ids_b[j])
            matched_dst.add(j)


def _window_link(
    graph: TrackingGraph,
    detections: dict[int, np.ndarray],
    ids_by_t: dict[int, list[int]],
    scale_arr: np.ndarray,
    params: LinkParams,
    descriptors: dict[int, np.ndarray] | None,
    edge_model,
    use_edge_model: bool,
) -> None:
    """Motion-coupled windowed global association (SOT-2871), in place.

    Processes consecutive ``t -> t+1`` transitions in time order. Each source's
    predicted position blends the SOT-2864 global motion field with a **carried
    velocity** running-averaged over the previous ``window_assoc - 1`` transitions
    (reset across any timepoint gap), so a middle detection's outgoing link cost
    depends on the incoming link the window chose one frame earlier — the cross-hop
    coupling that makes the short window bite. The per-transition assignment is a
    birth/death-outlier LAP (:func:`_window_assign`); optional parental-softmax adds
    balanced second daughters (:func:`_parental_softmax_divide`). Emits only
    consecutive-frame edges (metric-valid); short-track pruning and the division
    overlay are applied by the caller, as on the global path.
    """
    W = max(params.window_assoc, 2)
    theta = params.window_theta
    alpha = params.window_carry_weight
    gain = params.motion_gain
    sigma = params.motion_smooth_sigma
    use_desc = (params.appearance_weight > 0.0 or use_edge_model) and descriptors is not None

    times = sorted(detections)
    # Carried velocity (voxel space) into each matched detection, keyed by
    # (timepoint, local-index). Reset whenever the frame chain breaks.
    carry: dict[tuple[int, int], np.ndarray] = {}
    for t_a, t_b in zip(times, times[1:]):
        if t_b != t_a + 1:
            carry = {}  # gap: the window's motion history does not span missing frames
            continue
        src = detections[t_a]
        dst = detections[t_b]
        n = len(src)
        src_pred: np.ndarray | None = None
        if n and len(dst) and gain != 0.0:
            field_pred = _motion_field_predict(
                src, dst, scale_arr, params.max_distance, sigma, 1.0
            )
            field_disp = field_pred - src
            disp = field_disp.copy()
            for i in range(n):
                v = carry.get((t_a, i))
                if v is not None:
                    disp[i] = alpha * v + (1.0 - alpha) * field_disp[i]
            src_pred = src + gain * disp

        pairs = _window_assign(
            src, dst, scale_arr, params.max_distance, theta,
            src_pred=src_pred, disp_weight=params.velocity_disp_weight,
            gate_on_prediction=params.motion_gate_on_prediction,
            src_desc=descriptors[t_a] if use_desc else None,
            dst_desc=descriptors[t_b] if use_desc else None,
            appearance_weight=params.appearance_weight,
            edge_model=edge_model if use_edge_model else None,
        )

        # Update the carried velocity: a running mean over up to W-1 incoming hops,
        # so a larger window smooths the trajectory over more frames.
        new_carry: dict[tuple[int, int], np.ndarray] = {}
        for i, j in pairs:
            d = dst[j] - src[i]
            prev = carry.get((t_a, i))
            if prev is not None and W > 2:
                d = ((W - 2) * prev + d) / (W - 1)
            new_carry[(t_b, j)] = d
        carry = new_carry

        for i, j in pairs:
            graph.add_edge(ids_by_t[t_a][i], ids_by_t[t_b][j])

        if params.window_parental_softmax and params.allow_division and n and len(dst):
            _parental_softmax_divide(
                graph, ids_by_t[t_a], ids_by_t[t_b], src, dst, scale_arr, params, pairs
            )


def _kalman_link(
    graph: TrackingGraph,
    detections: dict[int, np.ndarray],
    ids_by_t: dict[int, list[int]],
    scale_arr: np.ndarray,
    params: LinkParams,
) -> None:
    """Per-track constant-velocity Kalman gating for the ``t -> t+1`` link (SOT-2920).

    Sequentially over consecutive-frame transitions, each track carries a
    constant-velocity Kalman filter (state ``[z, y, x, vz, vy, vx]`` in **scaled
    microns**). For the current source frame every track predicts its next position
    ``mu`` and *innovation covariance* ``S`` (a history-less source is initialised with a
    large :attr:`~LinkParams.kalman_init_vel_var`, so its gate is ~isotropic like the
    Euclidean champion). The assignment cost is the **Mahalanobis** distance
    ``sqrt((z-mu)ᵀ S⁻¹ (z-mu))`` and the feasibility gate is
    ``d_M² <= kalman_gate_chi2`` **intersected** with the champion's raw
    ``<= max_distance`` Euclidean cap — a pure restriction/re-rank of the champion
    feasible set (never admits a new long-range edge). Matched tracks are Kalman-updated
    and carried to the destination frame; unmatched destinations start fresh tracks.

    Adds only consecutive ``t -> t+1`` edges (division / gap knobs are inactive on this
    path, matching the champion's ``allow_division=false``); short-track pruning, the
    suspicious-review gate and the division overlay run afterwards in
    :func:`link_centroids` exactly as on the global/window paths. Pure numpy/scipy, CPU,
    offline, deterministic; mutates *graph* in place.
    """
    times = sorted(detections)
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = 1.0  # dt = 1 constant-velocity transition
    I3 = np.eye(3)
    I6 = np.eye(6)
    H = np.zeros((3, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0  # observe position only
    q = float(params.kalman_process_noise)
    r = max(float(params.kalman_obs_noise), 1e-9)
    Q = q * q * np.block([[0.25 * I3, 0.5 * I3], [0.5 * I3, I3]])
    R = r * r * I3
    P0 = np.zeros((6, 6))
    P0[:3, :3] = float(params.kalman_init_pos_var) * I3
    P0[3:, 3:] = float(params.kalman_init_vel_var) * I3
    chi2 = float(params.kalman_gate_chi2)
    max_d = float(params.max_distance)

    # carried[i] = (x[6], P[6,6]) for the track occupying source-index i of THIS frame.
    carried: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for t_a, t_b in zip(times, times[1:]):
        src = np.asarray(detections[t_a], dtype=float)
        dst = np.asarray(detections[t_b], dtype=float)
        ns, nd = len(src), len(dst)
        if t_b != t_a + 1 or ns == 0 or nd == 0:
            carried = {}  # temporal break: next consecutive frame starts fresh tracks
            continue
        src_s = src * scale_arr
        dst_s = dst * scale_arr
        # Assemble per-source Kalman state: fresh init (zero velocity, inflated P0)
        # overwritten by any track carried in from the previous transition.
        x = np.zeros((ns, 6))
        x[:, :3] = src_s
        P = np.repeat(P0[None, :, :], ns, axis=0)
        for i, (xi, Pi) in carried.items():
            if i < ns:
                x[i] = xi
                P[i] = Pi
        # Predict: xp = F x ; Pp = F P Fᵀ + Q  (batched over sources).
        xp = x @ F.T
        Pp = np.einsum("ab,ibc,dc->iad", F, P, F) + Q
        mu = xp[:, :3]  # (ns, 3) predicted positions
        S = Pp[:, :3, :3] + R  # (ns, 3, 3) innovation covariance
        Sinv = np.linalg.inv(S)
        diff = dst_s[None, :, :] - mu[:, None, :]  # (ns, nd, 3)
        maha2 = np.einsum("ijk,ikl,ijl->ij", diff, Sinv, diff)  # (ns, nd)
        raw = np.sqrt(((src_s[:, None, :] - dst_s[None, :, :]) ** 2).sum(axis=2))
        feasible = (raw <= max_d) & (maha2 <= chi2)
        maha = np.sqrt(np.maximum(maha2, 0.0))
        big = float(maha.max()) * 1000.0 + max_d * 1000.0 + 1.0
        cost = np.where(feasible, maha, big)
        rows, cols = linear_sum_assignment(cost)
        new_carried: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for a, b in zip(rows, cols):
            if not feasible[a, b]:
                continue
            # Kalman update of track ``a`` with observation ``dst_s[b]``.
            K = Pp[a] @ H.T @ Sinv[a]  # (6, 3) gain
            innov = dst_s[b] - mu[a]
            x_upd = xp[a] + K @ innov
            P_upd = (I6 - K @ H) @ Pp[a]
            new_carried[int(b)] = (x_upd, P_upd)
            graph.add_edge(ids_by_t[t_a][int(a)], ids_by_t[t_b][int(b)])
        carried = new_carried


def link_centroids(
    detections: dict[int, np.ndarray],
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    params: LinkParams | None = None,
    descriptors: dict[int, np.ndarray] | None = None,
) -> TrackingGraph:
    """Link per-timepoint centroids into a :class:`TrackingGraph`.

    ``detections`` maps ``timepoint -> (N, 3)`` centroid arrays in **voxel**
    coordinates. Node ids are assigned densely and deterministically (timepoint
    order, then detection order within a timepoint).

    ``descriptors`` (optional) maps ``timepoint -> (N, D)`` local appearance
    descriptors aligned row-for-row with ``detections`` (from
    :func:`biohub_tracking.detect.detect_volume_series_with_descriptors`). They are
    consulted only when ``params.appearance_weight > 0`` (SOT-2829); with the
    default ``appearance_weight == 0`` or ``descriptors is None`` the linking is the
    distance-only champion, **byte-for-byte**.
    """
    if params is None:
        params = LinkParams()
    scale_arr = np.asarray(scale, dtype=float)
    use_appearance = params.appearance_weight > 0.0 and descriptors is not None

    # GT-learned edge-linking cost (SOT-2841): built once from the embedded config.
    # Inactive (byte-for-byte champion) when no model, weight 0, or no descriptors.
    edge_model = None
    use_edge_model = False
    if params.edge_cost_model is not None and descriptors is not None:
        from .edge_linker import LearnedEdgeCost

        edge_model = LearnedEdgeCost.from_dict(params.edge_cost_model)
        use_edge_model = edge_model.weight != 0.0

    # Learned cross-attention edge-linking cost (SOT-2994): built once from the
    # embedded config. Inactive (byte-for-byte champion) when no model, weight 0, or
    # no descriptors. Inference is pure numpy (no torch at link time).
    xattn_model = None
    use_xattn_model = False
    if params.xattn_edge_model is not None and descriptors is not None:
        from .xattn_edge import CrossAttentionEdgeCost

        xattn_model = CrossAttentionEdgeCost.from_dict(params.xattn_edge_model)
        use_xattn_model = xattn_model.weight != 0.0

    use_desc = use_appearance or use_edge_model or use_xattn_model

    graph = TrackingGraph()
    ids_by_t: dict[int, list[int]] = {}
    next_id = 0
    for t in sorted(detections):
        ids: list[int] = []
        for z, y, x in detections[t]:
            graph.add_node(next_id, float(t), float(z), float(y), float(x))
            ids.append(next_id)
            next_id += 1
        ids_by_t[t] = ids

    # Whole-sequence Viterbi global track-linking with swaps (SOT-2918): a portable
    # port of Magnusson/Jaldén/Gilbert/Blau (IEEE TMI 2015). Re-solves the whole
    # sequence by iterated motion-coupled re-optimisation, so a link committed in one
    # sweep is swapped in a later sweep when whole-sequence evidence contradicts it —
    # the retroactive error-correction the greedy per-frame LAP lacks. Emits only
    # ``t -> t+1`` metric-valid edges; short-track pruning and the division overlay run
    # afterwards exactly as on the other global paths. ``viterbi_link`` False (default,
    # absent key) falls through to the unchanged champion path, so the champion graph
    # is byte-for-byte preserved.
    if params.viterbi_link:
        _viterbi_link(graph, detections, ids_by_t, scale_arr, params)
        if params.suspicious_review:
            graph = _suspicious_edge_review(
                graph, scale_arr, params.suspicious_turn_cos,
                params.suspicious_jump_ratio, params.suspicious_jump_floor,
            )
        if params.min_track_length > 1:
            graph = _prune_short_tracks(graph, params.min_track_length)
        if params.division_overlay:
            from .division_overlay import apply_division_overlay

            graph = apply_division_overlay(
                graph, tuple(scale_arr), params.division_overlay
            )
        return graph

    # Motion-coupled windowed global association (SOT-2871): a portable
    # Trackastra-style short-window LAP chain whose carried velocity couples adjacent
    # transitions (so the window bites, unlike the SOT-2830 pure-distance flow).
    # Supersedes ``global_window``; emits only ``t -> t+1`` metric-valid edges. Short-
    # track pruning and the division overlay run afterwards exactly as on the global
    # path. ``window_assoc <= 1`` (default) leaves the champion path untouched.
    if params.window_assoc > 1:
        _window_link(
            graph, detections, ids_by_t, scale_arr, params,
            descriptors, edge_model, use_edge_model,
        )
        if params.suspicious_review:
            graph = _suspicious_edge_review(
                graph, scale_arr, params.suspicious_turn_cos,
                params.suspicious_jump_ratio, params.suspicious_jump_floor,
            )
        if params.min_track_length > 1:
            graph = _prune_short_tracks(graph, params.min_track_length)
        if params.division_overlay:
            from .division_overlay import apply_division_overlay

            graph = apply_division_overlay(
                graph, tuple(scale_arr), params.division_overlay
            )
        return graph

    # Per-track constant-velocity Kalman gating (SOT-2920): each track carries its own
    # (position, velocity) state and forms an innovation-covariance-normalised
    # Mahalanobis gate, replacing/augmenting the fixed 7 µm Euclidean gate with a
    # per-track adaptive one. A dedicated sequential path; emits only ``t -> t+1``
    # metric-valid edges, then the shared post-processing (suspicious-review, short-track
    # prune, division overlay) runs exactly as on the global/window paths. Off (default)
    # falls through to the champion path below, byte-for-byte.
    if params.kalman_gate:
        _kalman_link(graph, detections, ids_by_t, scale_arr, params)
        if params.suspicious_review:
            graph = _suspicious_edge_review(
                graph, scale_arr, params.suspicious_turn_cos,
                params.suspicious_jump_ratio, params.suspicious_jump_floor,
            )
        if params.min_track_length > 1:
            graph = _prune_short_tracks(graph, params.min_track_length)
        if params.division_overlay:
            from .division_overlay import apply_division_overlay

            graph = apply_division_overlay(
                graph, tuple(scale_arr), params.division_overlay
            )
        return graph

    # Global short-window min-cost-flow linking (SOT-2830): explicit birth/death
    # arcs let the solver refuse a marginal link instead of greedily matching every
    # feasible pair. Only ``t -> t+1`` metric-valid edges are added (no bridges);
    # in-linker division and gap-closing are inactive on this path. ``global_window
    # <= 1`` (default) falls through to the unchanged per-frame champion path below,
    # so the champion graph is byte-for-byte preserved.
    if params.global_window > 1:
        _global_link(graph, detections, ids_by_t, scale_arr, params)
        if params.suspicious_review:
            graph = _suspicious_edge_review(
                graph, scale_arr, params.suspicious_turn_cos,
                params.suspicious_jump_ratio, params.suspicious_jump_floor,
            )
        if params.min_track_length > 1:
            graph = _prune_short_tracks(graph, params.min_track_length)
        if params.division_overlay:
            from .division_overlay import apply_division_overlay

            graph = apply_division_overlay(
                graph, tuple(scale_arr), params.division_overlay
            )
        return graph

    # Per-node incoming displacement, indexed by (timepoint, detection-index), so
    # a matched cell can predict where it is drifting next frame. Populated as
    # links are made; empty when ``velocity_gain == 0`` (memoryless champion path).
    times = sorted(detections)
    velocity_by_index: dict[tuple[int, int], np.ndarray] = {}
    for t_a, t_b in zip(times, times[1:]):
        if t_b != t_a + 1:
            continue  # only link consecutive timepoints
        src = detections[t_a]
        dst = detections[t_b]
        if params.local_affine_predict and len(src):
            # Local neighbour-affine prediction (SOT-2911): fit a per-cell affine
            # motion map to each source's k nearest provisional anchor displacements,
            # then LAP against the predicted positions. An independent axis from the
            # SOT-2864 global smoothed field — it captures the local velocity gradient
            # (shear/divergence) a single smoothed translation cannot.
            src_pred = _local_affine_predict(
                src, dst, scale_arr, params.max_distance,
                params.local_affine_k, params.local_affine_gain,
                params.local_affine_ridge,
            )
            pairs = _assign(
                src, dst, scale_arr, params.max_distance,
                src_pred=src_pred, disp_weight=params.velocity_disp_weight,
                gate_on_prediction=params.motion_gate_on_prediction,
                src_desc=descriptors[t_a] if use_desc else None,
                dst_desc=descriptors[t_b] if use_desc else None,
                appearance_weight=params.appearance_weight,
                edge_model=edge_model if use_edge_model else None,
                edge_gate_expand=params.edge_gate_expand,
                edge_gate_admit_prob=params.edge_gate_admit_prob,
                edge_gate_expand_ratio=params.edge_gate_expand_ratio,
                xattn_model=xattn_model if use_xattn_model else None,
            )
        elif params.motion_model_link and len(src):
            # ARGUS-style global-motion-field prediction (SOT-2864): predict every
            # source's next position from a smoothed field estimated *within this
            # frame pair*, then LAP against the predicted positions. Distinct from
            # the own-track velocity path below (predicts even history-less cells).
            src_pred = _motion_field_predict(
                src, dst, scale_arr, params.max_distance,
                params.motion_smooth_sigma, params.motion_gain,
            )
            # SOT-2883 bidirectional gate: the SAME global smoothed motion field
            # estimated on the *reversed* frame pair predicts where each dst came
            # from (backward). Built only when link_consistency_gate is set, so the
            # SOT-2864 path is byte-for-byte unchanged when off.
            dst_pred_bwd = None
            if params.link_consistency_gate and len(dst):
                dst_pred_bwd = _motion_field_predict(
                    dst, src, scale_arr, params.max_distance,
                    params.motion_smooth_sigma, params.motion_gain,
                )
            pairs = _assign(
                src, dst, scale_arr, params.max_distance,
                src_pred=src_pred, disp_weight=params.velocity_disp_weight,
                gate_on_prediction=params.motion_gate_on_prediction,
                src_desc=descriptors[t_a] if use_desc else None,
                dst_desc=descriptors[t_b] if use_desc else None,
                appearance_weight=params.appearance_weight,
                edge_model=edge_model if use_edge_model else None,
                edge_gate_expand=params.edge_gate_expand,
                edge_gate_admit_prob=params.edge_gate_admit_prob,
                edge_gate_expand_ratio=params.edge_gate_expand_ratio,
                xattn_model=xattn_model if use_xattn_model else None,
                dst_pred_bwd=dst_pred_bwd,
                consistency_gate=params.link_consistency_gate,
                consistency_tol=params.link_consistency_tol,
                consistency_weight=params.link_consistency_weight,
            )
        elif params.velocity_gain and len(src):
            src_pred = np.array(
                [
                    src[i] + params.velocity_gain * velocity_by_index.get((t_a, i), 0.0)
                    for i in range(len(src))
                ],
                dtype=float,
            )
            pairs = _assign(
                src, dst, scale_arr, params.max_distance,
                src_pred=src_pred, disp_weight=params.velocity_disp_weight,
                gate_on_prediction=params.motion_gate_on_prediction,
                src_desc=descriptors[t_a] if use_desc else None,
                dst_desc=descriptors[t_b] if use_desc else None,
                appearance_weight=params.appearance_weight,
                edge_model=edge_model if use_edge_model else None,
                edge_gate_expand=params.edge_gate_expand,
                edge_gate_admit_prob=params.edge_gate_admit_prob,
                edge_gate_expand_ratio=params.edge_gate_expand_ratio,
                xattn_model=xattn_model if use_xattn_model else None,
            )
        elif params.link_two_pass and not use_desc:
            # Classical-baseline two-pass tight-then-full-gate assignment (SOT-2899):
            # a pure feasibility-structure change on the memoryless champion path.
            # Pass 1 = the champion's tight max_distance assignment; Pass 2 recovers
            # leftovers out to link_full_distance. link_full_distance <= max_distance
            # (default) makes Pass 2 a no-op => champion byte-for-byte.
            pairs = _two_pass_assign(
                src, dst, scale_arr, params.max_distance, params.link_full_distance
            )
        else:
            pairs = _assign(
                src, dst, scale_arr, params.max_distance,
                src_desc=descriptors[t_a] if use_desc else None,
                dst_desc=descriptors[t_b] if use_desc else None,
                appearance_weight=params.appearance_weight,
                edge_model=edge_model if use_edge_model else None,
                edge_gate_expand=params.edge_gate_expand,
                edge_gate_admit_prob=params.edge_gate_admit_prob,
                edge_gate_expand_ratio=params.edge_gate_expand_ratio,
                xattn_model=xattn_model if use_xattn_model else None,
            )
        # Bidirectional mutual-NN cycle-consistency gate (SOT-2910): a pure
        # FP-edge suppressor that keeps only links agreeing in BOTH directions,
        # applied to the primary assignment before division / edge insertion.
        # Off (default) => champion byte-for-byte; it can only remove links, so
        # the <= max_distance cap and every recall-neutral edge are preserved.
        if params.cycle_consistency_gate:
            pairs = _cycle_consistency_filter(
                pairs, src, dst, scale_arr, params.cycle_consistency_margin
            )
        # Record each child's incoming displacement so it can predict t+1 -> t+2.
        if params.velocity_gain:
            for i, j in pairs:
                velocity_by_index[(t_b, j)] = dst[j] - src[i]
        matched_src = {i for i, _ in pairs}
        matched_dst = {j for _, j in pairs}
        for i, j in pairs:
            graph.add_edge(ids_by_t[t_a][i], ids_by_t[t_b][j])

        if not params.allow_division or len(src) == 0:
            continue
        # Scaled distance from each matched parent to its primary (assigned)
        # daughter, for the sibling-balance over-split gate.
        primary_dist: dict[int, float] = {}
        if params.division_max_sibling_ratio > 0.0:
            for i, j in pairs:
                primary_dist[i] = float(
                    np.sqrt((((src[i] - dst[j]) * scale_arr) ** 2).sum())
                )
        # Attach each leftover t+1 detection to its nearest matched parent as a
        # second daughter, if within division_distance (and, when the sibling
        # ratio gate is on, balanced against that parent's primary daughter).
        parent_pos = src * scale_arr
        for j in range(len(dst)):
            if j in matched_dst:
                continue
            d = np.sqrt(((parent_pos - dst[j] * scale_arr) ** 2).sum(axis=1))
            order = np.argsort(d)
            for i in order:
                if d[i] > params.division_distance:
                    break
                if i in matched_src and graph.out_degree(ids_by_t[t_a][i]) < 2:
                    if params.division_max_sibling_ratio > 0.0 and (
                        d[i] > params.division_max_sibling_ratio * primary_dist.get(i, 0.0)
                    ):
                        continue  # sibling too far vs primary daughter → not a split
                    graph.add_edge(ids_by_t[t_a][i], ids_by_t[t_b][j])
                    matched_dst.add(j)
                    break

    # Gap-closing 2nd LAP step runs BEFORE short-track pruning, so bridging real
    # short fragments into a >= min_track_length component rescues their internal
    # consecutive edges from the prune (SOT-2763).
    if params.max_frame_gap > 1:
        graph = _gap_close(
            graph, scale_arr, params.max_frame_gap, params.gap_distance
        )
    # Node-interpolation gap recovery (SOT-2849) also runs BEFORE pruning: it inserts
    # interpolated nodes on missing frames so the recovered path is *consecutive*
    # edges the metric scores (unlike the SOT-2763 bridge), and a recovered bridge
    # can lift two real short fragments into one surviving component.
    if params.gap_recover:
        graph = _gap_recover(
            graph,
            scale_arr,
            params.gap_recover_max_gap,
            params.gap_recover_distance,
            params.gap_recover_min_frag,
        )
    # Post-hoc suspicious-tracking-event review (SOT-2895) runs on the finished
    # linked graph, right BEFORE short-track pruning, so a cut decoy tail is left as
    # a short fragment the prune then removes (freeing the matched GT node and
    # shedding the FP edge). ``suspicious_review`` off → champion graph unchanged.
    if params.suspicious_review:
        graph = _suspicious_edge_review(
            graph, scale_arr, params.suspicious_turn_cos,
            params.suspicious_jump_ratio, params.suspicious_jump_floor,
        )
    if params.min_track_length > 1:
        graph = _prune_short_tracks(graph, params.min_track_length)
    # Non-destructive division overlay runs LAST, on the final pruned graph, so it
    # only adds second-daughter edges and never perturbs the linking assignment
    # (SOT-2818). ``None`` → champion graph unchanged, byte-for-byte.
    if params.division_overlay:
        from .division_overlay import apply_division_overlay

        graph = apply_division_overlay(graph, tuple(scale_arr), params.division_overlay)
    return graph
