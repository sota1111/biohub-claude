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

    # Base feasibility: the champion's raw (or, under gate_on_prediction, predicted)
    # distance gate.
    feasible = gate <= max_distance

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
    use_desc = use_appearance or use_edge_model

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
        if params.motion_model_link and len(src):
            # ARGUS-style global-motion-field prediction (SOT-2864): predict every
            # source's next position from a smoothed field estimated *within this
            # frame pair*, then LAP against the predicted positions. Distinct from
            # the own-track velocity path below (predicts even history-less cells).
            src_pred = _motion_field_predict(
                src, dst, scale_arr, params.max_distance,
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
    if params.min_track_length > 1:
        graph = _prune_short_tracks(graph, params.min_track_length)
    # Non-destructive division overlay runs LAST, on the final pruned graph, so it
    # only adds second-daughter edges and never perturbs the linking assignment
    # (SOT-2818). ``None`` → champion graph unchanged, byte-for-byte.
    if params.division_overlay:
        from .division_overlay import apply_division_overlay

        graph = apply_division_overlay(graph, tuple(scale_arr), params.division_overlay)
    return graph
