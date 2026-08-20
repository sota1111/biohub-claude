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
    smoothed = ndi.gaussian_filter(vol, sigma=params.sigma_zyx)

    # Detection response: raw smoothed intensity, or a Difference-of-Gaussians
    # (local contrast) response when a background sigma is configured (SOT-2272).
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
    return coords[order]


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
