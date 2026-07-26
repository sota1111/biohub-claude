"""Dependency-light OME-NGFF (zarr v3) reader for the detection pipeline.

The pipeline normally reads the ``*.zarr`` test videos with the ``zarr`` library,
but a Kaggle Code-competition kernel image may not ship ``zarr`` (and runs with no
internet), so the champion would be unable to read the pixels (SOT-1984). This
module reads a zarr-v3 ``(T, Z, Y, X)`` array directly from its on-disk chunks
using only ``numpy`` + a blosc decompressor (``numcodecs`` / ``blosc2`` /
``blosc`` — whichever is present), which are far more commonly available than
``zarr`` itself.

It supports the format the competition actually ships: a zarr-v3 array (or an
OME-NGFF multiscale group whose level ``0`` is the full-resolution array) with the
``bytes`` + ``blosc`` codec chain and the default ``/``-separated chunk keys. Only
the timepoint indexing the detector uses (``arr[t]`` → a 3D volume) is needed, so
chunks are decoded lazily, one timepoint at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _blosc_decompress(raw: bytes) -> bytes:
    """Decompress blosc-compressed *raw* bytes with whatever backend is present."""
    try:
        import numcodecs.blosc as _nb

        return _nb.decompress(raw)
    except ImportError:
        pass
    try:
        import blosc2

        return blosc2.decompress(raw)
    except ImportError:
        pass
    try:
        import blosc

        return blosc.decompress(raw)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "no blosc backend available (tried numcodecs, blosc2, blosc) to "
            "decode the OME-NGFF chunks without the zarr library"
        ) from exc


def _decompress_chunk(raw: bytes, codecs: list[dict]) -> bytes:
    """Apply the bytes->bytes codecs (only blosc is used here) in reverse."""
    names = [c.get("name") for c in codecs]
    if "blosc" in names:
        return _blosc_decompress(raw)
    if "gzip" in names:
        import gzip

        return gzip.decompress(raw)
    return raw  # 'bytes' codec only => stored raw


class NgffZarrV3Array:
    """Lazily read a zarr-v3 ``(T, Z, Y, X)`` array from its chunk files."""

    def __init__(self, array_dir: Path):
        self._dir = Path(array_dir)
        meta = json.loads((self._dir / "zarr.json").read_text())
        if meta.get("node_type") != "array":
            raise ValueError(f"{self._dir} is not a zarr v3 array")
        self.shape = tuple(int(s) for s in meta["shape"])
        self._chunks = tuple(
            int(s) for s in meta["chunk_grid"]["configuration"]["chunk_shape"]
        )
        endian = "<"
        for codec in meta.get("codecs", []):
            if codec.get("name") == "bytes":
                endian = ">" if codec["configuration"].get("endian") == "big" else "<"
        self._dtype = np.dtype(endian + np.dtype(meta["data_type"]).str[1:])
        self._codecs = meta.get("codecs", [])
        self._fill = meta.get("fill_value", 0)
        self._sep = (
            meta.get("chunk_key_encoding", {})
            .get("configuration", {})
            .get("separator", "/")
        )

    def _chunk_path(self, index: tuple[int, ...]) -> Path:
        key = self._sep.join(["c", *(str(i) for i in index)])
        return self._dir / key

    def _read_chunk(self, index: tuple[int, ...]) -> np.ndarray | None:
        path = self._chunk_path(index)
        if not path.exists():
            return None
        raw = _decompress_chunk(path.read_bytes(), self._codecs)
        return np.frombuffer(raw, dtype=self._dtype).reshape(self._chunks)

    def __getitem__(self, t: int) -> np.ndarray:
        """Return the full spatial volume ``(Z, Y, X)`` for timepoint *t*."""
        if not (0 <= t < self.shape[0]):
            raise IndexError(t)
        spatial = self.shape[1:]
        cz, cyy, cxx = self._chunks[1], self._chunks[2], self._chunks[3]
        t_chunk, t_off = divmod(t, self._chunks[0])
        out = np.full(spatial, self._fill, dtype=self._dtype)
        nz = -(-spatial[0] // cz)
        ny = -(-spatial[1] // cyy)
        nx = -(-spatial[2] // cxx)
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    chunk = self._read_chunk((t_chunk, iz, iy, ix))
                    if chunk is None:
                        continue
                    vol = chunk[t_off]  # (cz, cy, cx) for this timepoint
                    z0, y0, x0 = iz * cz, iy * cyy, ix * cxx
                    z1 = min(z0 + cz, spatial[0])
                    y1 = min(y0 + cyy, spatial[1])
                    x1 = min(x0 + cxx, spatial[2])
                    out[z0:z1, y0:y1, x0:x1] = vol[: z1 - z0, : y1 - y0, : x1 - x0]
        return out


def open_ome_ngff_array(zarr_path: Path | str) -> NgffZarrV3Array:
    """Open the full-resolution ``(T, Z, Y, X)`` array of an OME-NGFF ``*.zarr``.

    Handles both a bare zarr-v3 array and an OME multiscale group whose level
    ``0`` sub-array is the full-resolution image.
    """
    root = Path(zarr_path)
    level0 = root / "0"
    if (level0 / "zarr.json").exists():
        return NgffZarrV3Array(level0)
    if (root / "zarr.json").exists():
        meta = json.loads((root / "zarr.json").read_text())
        if meta.get("node_type") == "array":
            return NgffZarrV3Array(root)
    raise FileNotFoundError(f"no zarr v3 array found at {root} (or {root}/0)")
