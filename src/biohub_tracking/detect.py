"""Classical 3D cell detection on a light-sheet microscopy volume.

The detector is deliberately dependency-light (numpy + scipy only) so it runs
unchanged inside a Kaggle submission kernel. It treats each timepoint's
``(Z, Y, X)`` volume independently:

1. Smooth with an **anisotropic** Gaussian (a smaller sigma along ``Z`` because
   the voxels are ~4x coarser there), suppressing shot noise while keeping cell
   nuclei as compact bright blobs.
2. Find **local maxima** of the smoothed volume via a grey dilation
   (``maximum_filter``); the non-max-suppression footprint enforces a minimum
   separation between detections so one nucleus yields one centroid.
3. Keep only maxima brighter than a **percentile threshold** of the smoothed
   volume — an image-adaptive cutoff that survives the strong intensity drift
   across the developing embryo.

Everything is deterministic: the same volume and parameters always yield the
same centroids (no RNG), so scores are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

# Physical voxel size (z, y, x) in microns for the competition volumes.
DEFAULT_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class DetectParams:
    """Tunable knobs for :func:`detect_centroids` (all in voxel units)."""

    sigma_zyx: tuple[float, float, float] = (1.0, 3.0, 3.0)
    """Anisotropic Gaussian smoothing sigma (small along the coarse Z axis)."""

    nms_size_zyx: tuple[int, int, int] = (2, 5, 5)
    """Half-widths of the non-max-suppression window; the full footprint is
    ``2*size + 1`` along each axis, so this sets the minimum peak separation."""

    threshold_percentile: float = 99.5
    """Keep maxima brighter than this percentile of the **response** volume (the
    smoothed volume, or the DoG response when :attr:`background_sigma_zyx` is set)."""

    min_threshold: float = 0.0
    """Absolute floor on the response of a kept maximum."""

    background_sigma_zyx: tuple[float, float, float] | None = None
    """Optional Difference-of-Gaussians background sigma (SOT-2272). When set, the
    detection response is ``gaussian(sigma_zyx) - gaussian(background_sigma_zyx)``
    instead of the raw smoothed intensity, so peaks are found by **local contrast**
    rather than absolute brightness. This recovers dim cells that sit well below a
    global intensity percentile (the tracked cell in ``44b6_0b24845f`` sits at
    ~p60 of the smoothed volume and is invisible to a brightness threshold, but is
    a compact blob brighter than its immediate surround). ``None`` reproduces the
    original brightness-threshold detector exactly."""

    mad_k: float | None = None
    """Optional robust **adaptive** response threshold (SOT-2307). When set, the
    per-volume cutoff is ``median(response) + mad_k · 1.4826 · MAD(response)`` — a
    robust z-score above the response's own noise floor — instead of the fixed
    :attr:`threshold_percentile`. ``1.4826 · MAD`` (median absolute deviation) is an
    outlier-robust estimate of the response standard deviation, so the threshold
    tracks each dataset's *own* intensity scale rather than keeping a fixed voxel
    **fraction**. A fixed percentile keeps the same fraction of voxels regardless of
    how many real cells a volume actually contains, which massively over-detects
    sparse datasets (e.g. ``6bba_05b6850b``: the DoG p92 cutoff keeps ~40450 peaks
    for only ~6362 true cells, so the node-count penalty crushes its adjusted
    Jaccard to 0.26); an absolute robust cutoff keeps only genuine local-contrast
    peaks, adapting the detection *count* to signal content instead of volume size.
    ``None`` preserves the percentile behaviour exactly."""

    intensity_norm: tuple[str, float, float] | None = None
    """Optional **per-volume robust quantile** intensity normalization applied to
    the raw volume *before* any Gaussian smoothing (SOT-2776). When set to
    ``("quantile", plow, phigh)`` the volume is clipped to its own ``[plow, phigh]``
    percentile band and linearly rescaled to ``[0, 1]`` — the classical
    contrast-stretch royerlab's public baseline runs before detection. It puts
    every timepoint's intensities on a common ``[0, 1]`` scale so the per-volume
    threshold (percentile or MAD z-score) means the same thing across the strong
    brightness drift of the developing embryo, recovering dim family/timepoint
    cells that a raw-intensity cutoff drops. Only the *intensity* is normalized;
    anisotropy stays handled by :attr:`sigma_zyx`. ``None`` feeds the raw volume in
    unchanged (exact reproduction of the pre-SOT-2776 detector)."""

    dog_scales_zyx: tuple[tuple[float, float, float], ...] | None = None
    """Optional **multi-scale scale-space DoG** blob detection (SOT-2774).

    The single-scale DoG (:attr:`background_sigma_zyx`) tunes one blob radius, but
    real nuclei vary in size across family/timepoint (sparse ``44b6`` vs dense
    small-nuclei ``6bba``), so one scale both misses large cells and over-splits
    small ones. When set to a small bank of anisotropic foreground sigmas
    ``((σz,σy,σx), ...)`` (e.g. ``((0.7,1.4,1.4),(1.0,2.0,2.0),(1.4,2.8,2.8))``) the
    detector computes a DoG response at each scale and keeps the **per-voxel maximum
    normalized response across scales** — a scale-space blob detector in the spirit
    of skimage's ``blob_dog``/``blob_log`` but numpy/scipy-only (no new dependency).

    Each scale's background sigma is the base :attr:`background_sigma_zyx` scaled in
    proportion to its foreground sigma (using the base ``background/sigma`` ratio, so
    the middle scale that equals :attr:`sigma_zyx` reproduces the single-scale
    champion response); :attr:`background_sigma_zyx` must therefore be set. Each
    scale's DoG is multiplied by a **normalized-LoG scale correction** ``σ̄²`` (the
    squared geometric-mean sigma, ``prod(σ)^(2/3)``) so responses at different radii
    are directly comparable — without it the smaller scales, which sit on a steeper
    intensity gradient, would always dominate the max and defeat the point. The
    per-voxel argmax scale (which radius fired) is tracked internally. The combined
    response then feeds the *same* threshold (percentile or :attr:`mad_k` z-score)
    and NMS / watershed path unchanged. ``None`` runs the single-scale detector
    exactly (exact reproduction of the pre-SOT-2774 behaviour)."""

    watershed: tuple[str, float, float, float] | None = None
    """Optional **watershed nucleus splitting** with h-maxima seeding (SOT-2775).

    The NMS path (steps 2-3 above) reports one centroid per local maximum, so two
    nuclei whose blurred blobs merge into a single response peak collapse to one
    detection — the matched-edge FN/FP source on the *dense* ``6bba`` families
    (``6bba_05b6850b`` adj 0.5700 FN194 / ``6bba_05db0fb1`` adj 0.7310 FN210).
    When set to ``("hmaxima", h, min_size, min_seed_dist)`` a classical
    marker-controlled watershed is run *instead of* NMS on the same
    adaptive-threshold foreground: within each connected foreground component the
    **extended-maxima transform** (regional maxima of the h-maxima transform,
    ``scipy.ndimage`` grey reconstruction) yields seeds that survive a prominence
    of at least ``h`` robust-sigma (so shallow noise maxima do not seed a split),
    and the component is partitioned into one basin per seed by nearest-seed
    assignment (an exact seeded Euclidean region split — the marker-controlled
    watershed's deterministic geometric limit); the centroid of every basin is a
    detection. This splits fused blobs into their constituent nuclei without
    needing GPU weights.

    ``h`` is in units of the response's robust sigma (``1.4826·MAD``, exactly the
    :attr:`mad_k` scale), so the prominence gate means the same thing across the
    embryo's intensity drift. ``min_size`` is the minimum basin volume in **voxels**
    and ``min_seed_dist`` the minimum spacing between kept centroids in **voxels**
    — both suppress over-splitting (drop sub-``min_size`` basins; greedily keep the
    brightest of any centroids within ``min_seed_dist``). ``None`` runs the original
    NMS path unchanged (exact reproduction of the pre-SOT-2775 detector)."""

    blobness_filter: tuple[str, tuple[float, float, float], float, float] | None = None
    """Optional **Hessian-eigenvalue blobness precision filter** (SOT-2774 → SOT-2793).

    A detection *post-processing* gate: after NMS reports the candidate centroids it
    prunes any whose local structure is **not blob-like** (membrane / line / noise),
    to cut the sparse-family over-detection FPs that the node-count penalty punishes
    (``6bba_05b6850b`` over-detects) *without* re-tuning the response threshold. It is
    the mechanistic inverse of :attr:`dog_scales_zyx` (SOT-2774, which *adds*
    detections by a per-voxel scale-union max, REJECTED): here candidates are *removed*
    by a 3D blob-ness criterion from the HDoG / scale-space detection literature.

    When set to ``("hessian", sigma_zyx, min_blobness, max_anisotropy)`` the volume's
    **Hessian of Gaussian** (the six 2nd-derivative fields at anisotropic scale
    ``sigma_zyx``, via ``scipy.ndimage.gaussian_filter`` ``order=`` derivatives) is
    sampled at each candidate centroid and its three eigenvalues ``λ`` are taken. A
    candidate survives iff, ordering the eigenvalue **magnitudes** ``a0 ≤ a1 ≤ a2``:

    * **sign** — all three ``λ < 0`` (a bright blob in a dark background is a local
      intensity maximum in every direction, so every 2nd derivative is negative);
    * **isotropy** — ``a0 / a2 ≥ min_blobness`` (a *line/tube* has one eigenvalue ≈ 0
      along its axis, so ``a0/a2 → 0`` and it is dropped);
    * **planarity** — ``a2 / a1 ≤ max_anisotropy`` (a *plane/membrane* has two
      eigenvalues ≈ 0 in-plane, so ``a2/a1 → ∞`` and it is dropped).

    ``sigma_zyx`` is the (anisotropic) Gaussian derivative scale — reuse the base
    detection ``sigma_zyx`` so the Hessian is measured at the nucleus radius. The
    Hessian fields cost six extra full-volume Gaussian passes but the eigen-decomposition
    is only over the (few) candidate points, so it is cheap. The gate is applied only
    on the **NMS** path (the champion path); the watershed path is unaffected.
    ``None`` skips the filter entirely — exact byte-identical reproduction of the
    pre-SOT-2793 detector (default-off)."""


def _hessian_blobness_mask(
    vol: np.ndarray,
    coords: np.ndarray,
    spec: tuple[str, tuple[float, float, float], float, float],
) -> np.ndarray:
    """Boolean keep-mask: which candidate centroids are blob-like (SOT-2793).

    ``spec = ("hessian", sigma_zyx, min_blobness, max_anisotropy)``. Computes the
    Hessian of Gaussian (six 2nd-derivative fields at ``sigma_zyx``) of ``vol`` and
    samples it at each (rounded) candidate voxel in ``coords`` ``(N, 3)`` z,y,x. A
    candidate is kept iff its three Hessian eigenvalues are all negative (bright blob)
    and, with magnitudes sorted ``a0 ≤ a1 ≤ a2``, ``a0/a2 ≥ min_blobness`` (isotropic,
    not a line) and ``a2/a1 ≤ max_anisotropy`` (not a plane). Deterministic and
    numpy/scipy-only (Kaggle-kernel safe). Vectorised over candidates.
    """
    kind, sigma_zyx, min_blobness, max_anisotropy = spec
    if kind != "hessian":
        raise ValueError(f"unknown blobness_filter kind: {kind!r}")
    if coords.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    sigma = tuple(float(s) for s in sigma_zyx)
    min_blobness = float(min_blobness)
    max_anisotropy = float(max_anisotropy)

    # Hessian of Gaussian: the six unique 2nd-derivative fields (axes z=0, y=1, x=2).
    hzz = ndi.gaussian_filter(vol, sigma, order=(2, 0, 0))
    hyy = ndi.gaussian_filter(vol, sigma, order=(0, 2, 0))
    hxx = ndi.gaussian_filter(vol, sigma, order=(0, 0, 2))
    hzy = ndi.gaussian_filter(vol, sigma, order=(1, 1, 0))
    hzx = ndi.gaussian_filter(vol, sigma, order=(1, 0, 1))
    hyx = ndi.gaussian_filter(vol, sigma, order=(0, 1, 1))

    idx = np.rint(coords).astype(np.intp)
    for a in range(3):
        idx[:, a] = np.clip(idx[:, a], 0, vol.shape[a] - 1)
    z, y, x = idx[:, 0], idx[:, 1], idx[:, 2]

    n = coords.shape[0]
    h = np.empty((n, 3, 3), dtype=np.float64)
    h[:, 0, 0] = hzz[z, y, x]
    h[:, 1, 1] = hyy[z, y, x]
    h[:, 2, 2] = hxx[z, y, x]
    h[:, 0, 1] = h[:, 1, 0] = hzy[z, y, x]
    h[:, 0, 2] = h[:, 2, 0] = hzx[z, y, x]
    h[:, 1, 2] = h[:, 2, 1] = hyx[z, y, x]

    ev = np.linalg.eigvalsh(h)  # (N, 3) ascending
    all_negative = np.all(ev < 0.0, axis=1)
    mag = np.sort(np.abs(ev), axis=1)  # (N, 3) magnitudes a0 <= a1 <= a2
    a0, a1, a2 = mag[:, 0], mag[:, 1], mag[:, 2]
    isotropy = np.where(a2 > 0.0, a0 / a2, 0.0)
    planarity = np.where(a1 > 0.0, a2 / a1, np.inf)
    return (
        all_negative
        & (isotropy >= min_blobness)
        & (planarity <= max_anisotropy)
    )


def _normalize_intensity(
    vol: np.ndarray, spec: tuple[str, float, float] | None
) -> np.ndarray:
    """Apply the configured per-volume intensity normalization (SOT-2776).

    ``spec is None`` returns ``vol`` unchanged (exact reproduction). For
    ``("quantile", plow, phigh)`` the volume is clipped to its own ``[plow, phigh]``
    percentile band and linearly rescaled to ``[0, 1]``. Deterministic and
    numpy-only, so it runs unchanged inside a Kaggle kernel. A degenerate volume
    (constant band, ``hi <= lo``) is returned unchanged to avoid a divide-by-zero.
    """
    if spec is None:
        return vol
    kind = spec[0]
    if kind == "quantile":
        plow, phigh = float(spec[1]), float(spec[2])
        lo = float(np.percentile(vol, plow))
        hi = float(np.percentile(vol, phigh))
        if not hi > lo:  # constant/degenerate band — leave untouched
            return vol
        out = (vol - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    raise ValueError(f"unknown intensity_norm kind: {kind!r}")


def _multiscale_dog_response(
    vol: np.ndarray, params: "DetectParams"
) -> tuple[np.ndarray, np.ndarray]:
    """Scale-space DoG: per-voxel max normalized response over a sigma bank (SOT-2774).

    Returns ``(response, scale_index)`` — the per-voxel maximum normalized DoG
    response across :attr:`DetectParams.dog_scales_zyx`, and the index of the scale
    that fired at each voxel (which blob radius won; held for diagnostics/NMS parity
    with the single-scale path). Deterministic and numpy/scipy-only.

    Each scale ``σ`` uses a background sigma scaled from ``σ`` by the base
    ``background_sigma_zyx / sigma_zyx`` ratio, so the bank's middle scale (``== σ``)
    reproduces the single-scale champion DoG. Each DoG is multiplied by the
    **normalized-LoG scale correction** ``σ̄² = prod(σ)^(2/3)`` (squared geometric-mean
    sigma) so responses at different radii are comparable and the fine scales do not
    trivially dominate the max.
    """
    if params.background_sigma_zyx is None:
        raise ValueError(
            "dog_scales_zyx requires background_sigma_zyx (the base scale ratio)"
        )
    base_sigma = np.asarray(params.sigma_zyx, dtype=np.float64)
    base_bg = np.asarray(params.background_sigma_zyx, dtype=np.float64)
    # Elementwise background/foreground ratio from the base params; guard the
    # (impossible for real configs) zero-sigma axis so the ratio stays finite.
    ratio = np.where(base_sigma > 0.0, base_bg / np.where(base_sigma > 0.0, base_sigma, 1.0), 1.0)

    responses: list[np.ndarray] = []
    for scale in params.dog_scales_zyx:
        s = np.asarray(scale, dtype=np.float64)
        fg = ndi.gaussian_filter(vol, sigma=tuple(s))
        bg = ndi.gaussian_filter(vol, sigma=tuple(s * ratio))
        norm = float(np.prod(s)) ** (2.0 / 3.0)  # σ̄² normalized-LoG scale correction
        responses.append((fg - bg) * norm)

    stack = np.stack(responses, axis=0)  # (S, Z, Y, X)
    scale_index = np.argmax(stack, axis=0).astype(np.int16)
    response = np.max(stack, axis=0)
    return response, scale_index


def _reconstruction_by_dilation(
    marker: np.ndarray, mask: np.ndarray, structure: np.ndarray
) -> np.ndarray:
    """Grayscale morphological reconstruction by dilation (geodesic).

    Iterates ``min(dilate(marker), mask)`` to stability — the classical (slow but
    exact and deterministic) reconstruction. It is only ever called on a single
    foreground component's small bounding box, so the iteration count (bounded by
    the component diameter) is tiny and numpy/scipy-only (Kaggle-kernel safe).
    """
    prev = np.minimum(marker, mask)
    while True:
        cur = np.minimum(ndi.grey_dilation(prev, footprint=structure), mask)
        if np.array_equal(cur, prev):
            return cur
        prev = cur


def _extended_maxima(
    resp: np.ndarray, h: float, structure: np.ndarray
) -> np.ndarray:
    """Boolean seed mask = extended-maxima transform ``EMAX_h(resp)``.

    ``EMAX_h(f) = RMAX(HMAX_h(f))``: the regional maxima of the h-maxima transform,
    i.e. the maxima of ``resp`` whose prominence (dynamic) is at least ``h``. Shallow
    noise maxima (prominence ``< h``) are suppressed so they do not seed a spurious
    split. A component flatter than ``h`` everywhere collapses to a single seed
    plateau (one detection), so no foreground blob is ever lost.
    """
    hmax = _reconstruction_by_dilation(resp - h, resp, structure)
    rng = float(hmax.max() - hmax.min())
    if rng <= 0.0:  # fully flat component → the whole thing is one regional max
        return np.ones(hmax.shape, dtype=bool)
    eps = rng * 1e-6
    recon = _reconstruction_by_dilation(hmax - eps, hmax, structure)
    return (hmax - recon) > (eps * 0.5)


def _watershed_centroids(
    response: np.ndarray,
    foreground: np.ndarray,
    h: float,
    min_size: float,
    min_seed_dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split fused foreground blobs by marker-controlled watershed (SOT-2775).

    Returns ``(coords, strengths)`` — ``(M, 3)`` basin centroids in ``(z, y, x)``
    voxel coordinates and the peak response of each basin (for ordering). Operates
    per connected foreground component within its bounding box so the per-component
    reconstruction/watershed stays cheap. Deterministic and numpy/scipy-only.
    """
    conn = ndi.generate_binary_structure(3, 3)  # 26-connectivity, 3x3x3 ones
    labels, n = ndi.label(foreground, structure=conn)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    slices = ndi.find_objects(labels)

    coords_out: list[np.ndarray] = []
    strengths: list[float] = []
    for comp_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        comp_mask = labels[sl] == comp_id
        sub_resp = response[sl].astype(np.float64)
        offset = np.array([s.start for s in sl], dtype=np.float64)

        # Seeds: extended maxima (prominence >= h) within the component only. Fill
        # the bbox's non-component voxels with the component floor (a finite low
        # value) so the reconstruction sees them as background — they cannot seed a
        # maximum and are dropped by ``& comp_mask`` — while avoiding non-finite math.
        floor = float(sub_resp[comp_mask].min())
        masked = np.where(comp_mask, sub_resp, floor)
        seeds = _extended_maxima(masked, h, conn) & comp_mask
        seed_labels, n_seeds = ndi.label(seeds, structure=conn)

        if n_seeds <= 1:
            # No fusion to split — one detection at the component's brightest voxel.
            basins = [(comp_mask, float(sub_resp[comp_mask].max()))]
        else:
            # Marker-controlled split: assign every foreground voxel to its nearest
            # seed (an exact Euclidean seeded region partition via the distance
            # transform's feature indices). For compact nuclei the partition falls
            # at the geometric midline between the h-maxima seeds — the response
            # ridge separating the fused blobs — and is fully deterministic (no
            # RNG, no order-dependent flooding), splitting the component into one
            # basin per seed.
            _, (iz, iy, ix) = ndi.distance_transform_edt(
                seed_labels == 0, return_indices=True
            )
            assigned = np.where(comp_mask, seed_labels[iz, iy, ix], 0)
            basins = []
            for lbl in range(1, n_seeds + 1):
                region = assigned == lbl
                if region.any():
                    basins.append((region, float(sub_resp[region].max())))

        for region, strength in basins:
            vol_vox = int(region.sum())
            if vol_vox < min_size:
                continue
            pts = np.argwhere(region).astype(np.float64) + offset
            coords_out.append(pts.mean(axis=0))
            strengths.append(strength)

    if not coords_out:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    coords = np.vstack(coords_out)
    strong = np.asarray(strengths, dtype=np.float64)

    # Order brightest-first, then greedily suppress centroids closer than
    # ``min_seed_dist`` voxels (over-split control; keeps the brightest).
    order = np.argsort(strong)[::-1]
    coords, strong = coords[order], strong[order]
    if min_seed_dist > 0.0 and len(coords) > 1:
        keep = np.ones(len(coords), dtype=bool)
        d2 = min_seed_dist * min_seed_dist
        for i in range(len(coords)):
            if not keep[i]:
                continue
            diff = coords[i + 1 :] - coords[i]
            close = (diff * diff).sum(axis=1) <= d2
            keep[i + 1 :][close & keep[i + 1 :]] = False
        coords, strong = coords[keep], strong[keep]
    return coords, strong


def detect_centroids(
    volume: np.ndarray, params: DetectParams | None = None
) -> np.ndarray:
    """Detect cell centroids in one 3D ``(Z, Y, X)`` volume.

    Returns an ``(N, 3)`` float array of ``(z, y, x)`` centroids in **voxel**
    coordinates, ordered by descending smoothed intensity (brightest first) so
    that any downstream node-count cap keeps the most confident detections.
    """
    if params is None:
        params = DetectParams()

    vol = np.asarray(volume, dtype=np.float32)
    vol = _normalize_intensity(vol, params.intensity_norm)

    # Detection response, in precedence order:
    #   1. multi-scale scale-space DoG (per-voxel max over a sigma bank; SOT-2774),
    #   2. single-scale Difference-of-Gaussians local contrast (SOT-2272), or
    #   3. the raw smoothed intensity.
    if params.dog_scales_zyx is not None:
        response, _scale_index = _multiscale_dog_response(vol, params)
    else:
        smoothed = ndi.gaussian_filter(vol, sigma=params.sigma_zyx)
        if params.background_sigma_zyx is not None:
            background = ndi.gaussian_filter(vol, sigma=params.background_sigma_zyx)
            response = smoothed - background
        else:
            response = smoothed

    # Response threshold: a robust per-volume z-score (median + k·1.4826·MAD) when
    # ``mad_k`` is configured (SOT-2307, adapts the kept-peak *count* to each
    # dataset's own noise floor), otherwise the fixed percentile cutoff (SOT-2272).
    if params.mad_k is not None:
        median = float(np.median(response))
        mad = float(np.median(np.abs(response - median)))
        robust_sigma = 1.4826 * mad
        adaptive_threshold = median + params.mad_k * robust_sigma
    else:
        adaptive_threshold = float(np.percentile(response, params.threshold_percentile))
    threshold = max(adaptive_threshold, params.min_threshold)

    # Watershed splitting path (SOT-2775): replace NMS peak extraction with
    # marker-controlled watershed over the same adaptive-threshold foreground, so
    # fused nuclei that share one response peak are split into per-basin centroids.
    if params.watershed is not None:
        kind, h_sigma, min_size, min_seed_dist = params.watershed
        if kind != "hmaxima":
            raise ValueError(f"unknown watershed kind: {kind!r}")
        foreground = response > threshold
        if not foreground.any():
            return np.zeros((0, 3), dtype=np.float64)
        # h is in robust-sigma units (1.4826·MAD of the response), so the
        # prominence gate is comparable across volumes regardless of intensity scale.
        median = float(np.median(response))
        mad = float(np.median(np.abs(response - median)))
        h_abs = float(h_sigma) * 1.4826 * mad
        coords, _strong = _watershed_centroids(
            response, foreground, h_abs, float(min_size), float(min_seed_dist)
        )
        return coords

    footprint = np.ones([2 * s + 1 for s in params.nms_size_zyx], dtype=bool)
    local_max = ndi.maximum_filter(response, footprint=footprint)
    peak_mask = (response == local_max) & (response > threshold)

    coords = np.argwhere(peak_mask).astype(np.float64)  # (N, 3) z,y,x
    if coords.size == 0:
        return coords.reshape(0, 3)

    intensities = response[peak_mask]
    order = np.argsort(intensities)[::-1]
    coords = coords[order]

    # Hessian-eigenvalue blobness precision filter (SOT-2793): prune non-blob
    # (membrane/line/noise) candidates as a detection post-processing step. Runs
    # only when configured, so the champion (blobness_filter=None) is untouched.
    if params.blobness_filter is not None:
        keep = _hessian_blobness_mask(vol, coords, params.blobness_filter)
        coords = coords[keep]

    return coords


def detect_volume_series(
    zarr_array, params: DetectParams | None = None, max_t: int | None = None
) -> dict[int, np.ndarray]:
    """Detect centroids for every timepoint of a ``(T, Z, Y, X)`` zarr array.

    ``zarr_array`` may be any object indexable as ``arr[t]`` returning a 3D
    volume (a :mod:`zarr` array or an in-memory ndarray). Returns
    ``{t: (N, 3) centroids}`` in voxel coordinates.
    """
    n_t = zarr_array.shape[0] if max_t is None else min(max_t, zarr_array.shape[0])
    return {t: detect_centroids(zarr_array[t], params) for t in range(n_t)}
