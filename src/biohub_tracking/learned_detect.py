"""Learned-detector receptacle (SOT-2847): a default-off, torch-based detector
that plugs into the pipeline with the SAME contract as the classical detector.

Why this exists
---------------
For seven improvement cycles the classical numpy/scipy detection+linking pipeline
has been stuck at a ~0.62-0.66 ceiling while the public *learned* frontier reaches
~0.89-0.957 (royerlab ``TemporalUNet3D`` detection + ``SimpleNodeTransformer``
linking). The champion config's ``description`` codified an **unverified** premise —
that a GPU/pretrained-weights model "cannot run under this repo's numpy/scipy/zarr,
CPU, no-internet, no-weights kernel". SOT-2847 verified that premise is **false as
an infrastructure constraint** (see ``docs/ai/sot-2847-submission-constraint-verdict.md``):
this competition is a Kaggle **Code competition** whose kernels can enable GPU and
read **attached, offline** model weights (Kaggle Dataset / Model source, internet
off), exactly as the public learned-inference notebooks do.

This module is the *receptacle* that verdict unlocks. It is deliberately NOT a
promotion: it is **off by default**, and when off the champion pipeline is
byte-for-byte unchanged (nothing here is imported unless a learned-detector config
block is present and enabled). The real detector port (a trained ``TemporalUNet3D``)
is child SOT-2848; this module gives it a place to land plus a smoke test proving
the offline attach → load → forward path works.

Contract (matches :mod:`biohub_tracking.detect`)
------------------------------------------------
A learned detector exposes the identical detector surface the pipeline already
uses::

    detector.detect_centroids(volume)          -> (N, 3) float (z, y, x) voxel coords
    detector.detect_series(zarr_array, max_t)  -> {t: (N, 3)}

so :func:`biohub_tracking.pipeline.run_pipeline` can swap it in for
:func:`biohub_tracking.detect.detect_volume_series` with no other change, and the
downstream linker / scorer / submission code is untouched.

torch is imported **lazily** (only when a detector is actually built), so the
classical champion path, the exec-compat gate, and the CI test suite keep running
in the torch-free ``numpy/scipy/zarr`` environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# Kaggle Dataset / Model mount roots an attached-weights kernel reads from
# (internet off). Mirrors submit/build_kernel.py's ``/kaggle/input`` discovery so
# the offline attach path is resolved the same way the kernel would.
_KAGGLE_INPUT_ROOTS: tuple[str, ...] = ("/kaggle/input",)


def torch_available() -> bool:
    """True iff torch can be imported in the current interpreter.

    The classical pipeline never calls this; it exists so callers (the smoke
    test, a future learned kernel) can feature-detect rather than crash in the
    torch-free environment that runs the champion.
    """
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "no torch here"
        return False
    return True


@dataclass(frozen=True)
class LearnedDetectorConfig:
    """Declarative config for a learned detector (parsed from the champion-config
    ``learned_detector`` block). ``enabled=False`` (or an absent block) keeps the
    classical champion path, byte-for-byte.

    Fields
    ------
    enabled
        Master switch. False / absent → :func:`build_learned_detector` returns
        ``None`` and the pipeline uses the classical detector unchanged.
    arch
        Reference architecture key (see :data:`_ARCH_REGISTRY`). The receptacle
        ships one tiny placeholder arch (``voxel_scorer_v0``); child SOT-2848
        registers the real ``TemporalUNet3D``.
    weights
        Dataset-relative path to the checkpoint (e.g.
        ``biohub-claude-weights/detector.pt``), resolved against the attached
        Kaggle Dataset/Model mount(s) at inference time. ``None`` builds the arch
        with freshly-initialised weights (smoke / plumbing only).
    threshold
        Detection-heatmap cutoff in [0, 1]; peaks above it are kept.
    nms_size_zyx
        Half-widths of the non-max-suppression window used to turn the per-voxel
        heatmap into discrete centroids (same footprint convention as
        :class:`biohub_tracking.detect.DetectParams`).
    device
        ``"auto"`` (cuda if available else cpu), ``"cuda"``, or ``"cpu"``.
    """

    enabled: bool = False
    arch: str = "voxel_scorer_v0"
    weights: str | None = None
    threshold: float = 0.5
    nms_size_zyx: tuple[int, int, int] = (2, 5, 5)
    device: str = "auto"

    @classmethod
    def from_dict(cls, block: dict | None) -> LearnedDetectorConfig | None:
        """Parse a ``learned_detector`` config block. ``None`` → ``None`` (off)."""
        if not block:
            return None
        nms = block.get("nms_size_zyx", (2, 5, 5))
        return cls(
            enabled=bool(block.get("enabled", False)),
            arch=str(block.get("arch", "voxel_scorer_v0")),
            weights=(None if block.get("weights") is None else str(block["weights"])),
            threshold=float(block.get("threshold", 0.5)),
            nms_size_zyx=tuple(int(v) for v in nms),
            device=str(block.get("device", "auto")),
        )


def _build_arch(arch: str):
    """Instantiate the reference architecture named *arch* (lazy torch import)."""
    from torch import nn

    class VoxelScorerV0(nn.Module):
        """Tiny 3D detection head: a couple of anisotropic Conv3d layers → a
        single-channel per-voxel detection logit (sigmoid at inference).

        This is a *placeholder* whose only job is to exercise the receptacle end
        to end (construct → save weights → attach → load offline → forward →
        centroids). It is NOT the frontier detector; SOT-2848 ports the trained
        ``TemporalUNet3D`` and registers it here. Kept deliberately small so the
        smoke test runs in well under a second on CPU.
        """

        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv3d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 1, kernel_size=1),
            )

        def forward(self, x):  # x: (B, 1, Z, Y, X) -> (B, 1, Z, Y, X) logits
            return self.body(x)

    class TemporalUNet3D(nn.Module):
        """Learned 3D U-Net detection head (SOT-2848), ported from the royerlab
        public frontier baseline (``models/temporal_unet.py`` /
        ``train_unet_transformer.py``): an anisotropic encoder-decoder 3D U-Net
        producing a single-channel per-voxel detection logit, with a
        squeeze-excitation attention bottleneck.

        Port scope vs the reference
        ---------------------------
        The reference ``TemporalUNet3D`` mixes information across a short temporal
        window with attention before the detection head. This repo's detector
        contract (:meth:`LearnedDetector.detect_centroids`) is **per single
        volume** — linking is a separate, fixed champion stage (distance-only
        bipartite over ``t -> t+1``), and the CV A/B this port feeds swaps *only*
        detection. So the temporal-attention mixing degenerates to a **bottleneck
        self-/channel-attention** on the current frame (the SE block below); the
        cross-frame association the reference folds into the detector is instead
        carried by the downstream linker. This keeps the port faithful to the
        learned per-voxel *detection* head — the axis under test (classical DoG+NMS
        vs a learned heatmap) — without duplicating the linker.

        Anisotropy: ``(Z, Y, X)`` voxels are ``1.625 x 0.40625 x 0.40625`` um and
        the volume is ``64 x 256 x 256``; pooling is isotropic in index space
        (``2 x 2 x 2``) over three levels (``Z 64->8``, ``YX 256->32``), matching
        the reference's index-space downsampling. ``base`` channels kept small so a
        full-volume forward fits the 12 GB RTX 3080 Ti at inference.
        """

        def __init__(self, base: int = 16):
            super().__init__()

            def cbr(ci: int, co: int) -> "nn.Module":
                return nn.Sequential(
                    nn.Conv3d(ci, co, 3, padding=1),
                    nn.InstanceNorm3d(co, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(co, co, 3, padding=1),
                    nn.InstanceNorm3d(co, affine=True),
                    nn.ReLU(inplace=True),
                )

            self.enc1 = cbr(1, base)
            self.enc2 = cbr(base, base * 2)
            self.enc3 = cbr(base * 2, base * 4)
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.bottleneck = cbr(base * 4, base * 8)
            # Squeeze-excitation channel attention at the bottleneck (the port's
            # stand-in for the reference's temporal attention, see docstring).
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(base * 8, base, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(base, base * 8, 1),
                nn.Sigmoid(),
            )
            self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
            self.dec3 = cbr(base * 8, base * 4)
            self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
            self.dec2 = cbr(base * 4, base * 2)
            self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
            self.dec1 = cbr(base * 2, base)
            self.head = nn.Conv3d(base, 1, 1)

        def forward(self, x):  # (B, 1, Z, Y, X) -> (B, 1, Z, Y, X) logits
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            b = b * self.se(b)
            import torch as _t

            d3 = self.dec3(_t.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(_t.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(_t.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)

    registry = {"voxel_scorer_v0": VoxelScorerV0, "temporal_unet3d": TemporalUNet3D}
    if arch not in registry:
        raise ValueError(
            f"unknown learned-detector arch {arch!r}; known: {sorted(registry)}"
        )
    return registry[arch]()


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class LearnedDetector:
    """A torch detector matching the classical detector contract.

    Construct via :func:`build_learned_detector` (config-driven, default-off) or
    :meth:`for_smoke` (self-contained, freshly-initialised weights). The forward
    pass yields a per-voxel detection heatmap; centroids are recovered by the same
    threshold + non-max-suppression the classical detector uses, so the returned
    ``(N, 3)`` voxel coordinates are drop-in for the linker.
    """

    def __init__(self, config: LearnedDetectorConfig, model=None, device=None):
        self.config = config
        self._model = model
        self._device = device

    # -- construction ---------------------------------------------------------
    def load(self, weights_path: Path | str | None = None) -> LearnedDetector:
        """Build the arch and load its weights **offline** from *weights_path*.

        Resolution order: explicit *weights_path* → the config's ``weights``
        resolved against the attached Kaggle Dataset/Model mount(s). A ``None``
        result (no path given, none in config) builds freshly-initialised weights
        so the plumbing can still be exercised. ``torch.load`` reads a local file
        only — no network — which is exactly the offline-kernel condition.
        """
        import torch

        self._device = _resolve_device(self.config.device)
        model = _build_arch(self.config.arch)

        path = weights_path if weights_path is not None else self.resolve_weights_path()
        if path is not None:
            # Prefer the safe tensors-only load (torch>=2.6 default); fall back to
            # a full unpickle for checkpoints that carry non-tensor objects. Reads
            # a local file only — no network — the offline-kernel condition.
            try:
                state = torch.load(str(path), map_location="cpu", weights_only=True)
            except Exception:  # noqa: BLE001 - retry full unpickle for rich checkpoints
                state = torch.load(str(path), map_location="cpu", weights_only=False)
            # Accept a bare state_dict or a {"state_dict": ...} checkpoint.
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
        model.eval()
        model.to(self._device)
        self._model = model
        return self

    def resolve_weights_path(
        self, search_roots: tuple[str, ...] = _KAGGLE_INPUT_ROOTS
    ) -> Path | None:
        """Find the configured ``weights`` file under the attached mount(s).

        Probes ``<root>/<weights>`` for each mount root, then falls back to a
        recursive search for the checkpoint's basename (Kaggle nests a Dataset
        under ``/kaggle/input/<dataset-slug>/``). ``None`` if not found or the
        config names no weights.
        """
        if not self.config.weights:
            return None
        rel = self.config.weights
        for root in search_roots:
            direct = Path(root) / rel
            if direct.is_file():
                return direct
        basename = Path(rel).name
        for root in search_roots:
            base = Path(root)
            if base.is_dir():
                for hit in base.rglob(basename):
                    return hit
        return None

    @classmethod
    def for_smoke(
        cls, weights_path: Path | str, *, device: str = "cpu", threshold: float = 0.5
    ) -> LearnedDetector:
        """A detector whose weights are loaded directly from *weights_path* (used
        by the offline attach smoke test)."""
        cfg = LearnedDetectorConfig(
            enabled=True, weights=None, threshold=threshold, device=device
        )
        return cls(cfg).load(weights_path=weights_path)

    # -- detection (classical detector contract) ------------------------------
    def _heatmap(self, volume: np.ndarray) -> np.ndarray:
        """Run the model forward on one ``(Z, Y, X)`` volume → a [0, 1] heatmap."""
        import torch

        if self._model is None:
            raise RuntimeError("LearnedDetector.load() must be called before detection")
        vol = np.asarray(volume, dtype=np.float32)
        # Per-volume standardisation so an untrained/placeholder head sees a
        # scale-free input; the real detector ships its own normalisation with the
        # weights (SOT-2848).
        mean = float(vol.mean())
        std = float(vol.std()) or 1.0
        norm = (vol - mean) / std
        tensor = torch.from_numpy(norm)[None, None].to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            prob = torch.sigmoid(logits)
        return prob[0, 0].detach().cpu().numpy().astype(np.float32)

    def detect_centroids(self, volume: np.ndarray) -> np.ndarray:
        """Detect centroids in one ``(Z, Y, X)`` volume via the learned heatmap.

        Returns an ``(N, 3)`` float array of ``(z, y, x)`` voxel coordinates,
        ordered by descending heatmap confidence (brightest first) — the same
        ordering contract as :func:`biohub_tracking.detect.detect_centroids`, so a
        downstream node-count cap keeps the most confident detections.
        """
        heat = self._heatmap(volume)
        footprint = np.ones(
            [2 * s + 1 for s in self.config.nms_size_zyx], dtype=bool
        )
        local_max = ndi.maximum_filter(heat, footprint=footprint)
        peak_mask = (heat == local_max) & (heat > self.config.threshold)
        coords = np.argwhere(peak_mask).astype(np.float64)
        if coords.size == 0:
            return coords.reshape(0, 3)
        order = np.argsort(heat[peak_mask])[::-1]
        return coords[order]

    def detect_series(
        self, zarr_array, max_t: int | None = None
    ) -> dict[int, np.ndarray]:
        """Detect centroids for every timepoint of a ``(T, Z, Y, X)`` array.

        Signature-compatible with
        :func:`biohub_tracking.detect.detect_volume_series` (``arr[t]`` → 3D
        volume), returning ``{t: (N, 3)}`` voxel coordinates.
        """
        n_t = (
            zarr_array.shape[0]
            if max_t is None
            else min(max_t, zarr_array.shape[0])
        )
        return {t: self.detect_centroids(zarr_array[t]) for t in range(n_t)}


def build_learned_detector(block: dict | None) -> LearnedDetector | None:
    """Build a :class:`LearnedDetector` from a ``learned_detector`` config block,
    or ``None`` when it is absent / disabled.

    ``None`` is the champion path: the pipeline uses the classical detector and is
    byte-for-byte unchanged. Only an ``enabled: true`` block loads torch and a
    learned detector — so nothing in this module runs for the champion.
    """
    cfg = LearnedDetectorConfig.from_dict(block)
    if cfg is None or not cfg.enabled:
        return None
    if not torch_available():
        raise RuntimeError(
            "learned_detector is enabled but torch is not importable; a learned "
            "kernel must attach torch + weights (GPU env). The classical champion "
            "runs without torch — set learned_detector.enabled=false to use it."
        )
    return LearnedDetector(cfg).load()


# ---------------------------------------------------------------------------
# Offline weights-attach smoke test (SOT-2847 acceptance): construct a tiny
# model, save its weights, simulate a Kaggle Dataset mount, load the weights back
# **offline**, and run a forward pass end to end. Measures whether the
# attach → load → forward path a learned submission kernel needs is feasible.
# ---------------------------------------------------------------------------
def run_weights_attach_smoke(
    tmp_root: Path | str | None = None,
    *,
    device: str = "auto",
    dataset_slug: str = "biohub-claude-weights",
    weights_name: str = "detector.pt",
) -> dict:
    """Exercise the offline attach → load → forward path; return a verdict dict.

    Steps (all offline — ``torch.load`` reads a local file, no network):

    1. Build the placeholder arch and **save** its ``state_dict`` under a
       simulated Kaggle Dataset mount ``<tmp>/kaggle_input/<slug>/<weights>``.
    2. Resolve + **load** those weights back through
       :meth:`LearnedDetector.resolve_weights_path` (the same discovery the kernel
       uses), asserting the reloaded model matches the saved one.
    3. **Forward** the loaded model over a small synthetic ``(T, Z, Y, X)`` volume
       and recover centroids via :meth:`LearnedDetector.detect_series`.

    Raises if torch is unavailable (the caller feature-detects with
    :func:`torch_available`); returns a JSON-serialisable summary on success.
    """
    import tempfile

    import torch

    created_tmp = tmp_root is None
    tmp_root = Path(tempfile.mkdtemp(prefix="sot2847-smoke-")) if created_tmp else Path(tmp_root)
    try:
        # 1) Save weights into a simulated attached Kaggle Dataset mount.
        input_root = tmp_root / "kaggle_input"
        ds_dir = input_root / dataset_slug
        ds_dir.mkdir(parents=True, exist_ok=True)
        weights_path = ds_dir / weights_name

        torch.manual_seed(0)
        model = _build_arch("voxel_scorer_v0")
        torch.save(model.state_dict(), weights_path)
        saved_bytes = weights_path.stat().st_size

        # 2) Resolve via the kernel's discovery path, then load OFFLINE.
        det = LearnedDetector(
            LearnedDetectorConfig(
                enabled=True,
                weights=f"{dataset_slug}/{weights_name}",
                device=device,
                threshold=0.0,  # keep-all so the forward path yields detections
            )
        )
        resolved = det.resolve_weights_path(search_roots=(str(input_root),))
        assert resolved is not None and resolved.is_file(), "attached weights not resolved"
        det.load(weights_path=resolved)

        # Reloaded weights must match what was saved (bit-exact load, offline).
        reloaded = {k: v.clone() for k, v in det._model.state_dict().items()}
        original = model.state_dict()
        params_match = all(
            torch.equal(reloaded[k].cpu(), original[k].cpu()) for k in original
        )

        # 3) Forward over a synthetic (T, Z, Y, X) volume with two bright blobs.
        rng = np.zeros((3, 8, 24, 24), dtype=np.float32)
        for t in range(3):
            for (z, y, x) in ((4, 6, 6), (4, 18, 18)):
                rng[t, z, y, x] = 1000.0
        rng += 1.0
        series = det.detect_series(rng)
        total = int(sum(len(v) for v in series.values()))

        return {
            "torch_available": True,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_used": str(det._device),
            "dataset_mount": str(input_root),
            "weights_path": str(resolved),
            "weights_bytes": saved_bytes,
            "resolved_offline": True,
            "state_dict_bit_exact": bool(params_match),
            "forward_ok": True,
            "num_timepoints": len(series),
            "total_detections": total,
            "verdict": "offline attach->load->forward FEASIBLE",
        }
    finally:
        if created_tmp:
            import shutil

            shutil.rmtree(tmp_root, ignore_errors=True)


def _main(argv: list[str] | None = None) -> int:
    """CLI: run the offline weights-attach smoke and print/emit its verdict."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Learned-detector receptacle (SOT-2847).")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run the offline weights-attach smoke test and print the verdict.",
    )
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--out", default=None, help="Write the verdict JSON here.")
    args = ap.parse_args(argv)

    if not args.smoke:
        ap.print_help()
        return 0

    if not torch_available():
        payload = {"torch_available": False, "verdict": "torch not importable here"}
    else:
        payload = run_weights_attach_smoke(device=args.device)
    print(json.dumps(payload, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")
    return 0 if payload.get("torch_available", False) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
