"""Tests for the dependency-light OME-NGFF reader (SOT-1984).

``biohub_tracking.ngff`` reads a zarr-v3 ``(T, Z, Y, X)`` array directly from its
chunk files using only numpy + a blosc backend, so the champion can read the
competition videos inside a Kaggle Code-competition kernel whose image may not
ship the ``zarr`` library. These tests build a tiny array on disk in the *exact*
format the competition ships (``bytes`` + ``blosc`` codecs, one full timepoint
per chunk) and assert the reader reproduces it byte-for-byte and that the
pipeline transparently falls back to it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from biohub_tracking import pipeline as pipeline_mod
from biohub_tracking.ngff import open_ome_ngff_array


def _blosc_compress(buf: bytes) -> bytes:
    """Compress with whatever blosc backend the ngff reader can decode."""
    try:
        import numcodecs

        return numcodecs.Blosc(cname="zstd", clevel=1).encode(buf)
    except ImportError:  # pragma: no cover - environment dependent
        import blosc

        return blosc.compress(buf, typesize=1)


def _write_competition_zarr(array_dir, data: np.ndarray):
    """Write *data* (T,Z,Y,X) as a zarr-v3 array matching the competition format.

    One chunk per timepoint (chunk_shape ``(1, Z, Y, X)``), ``bytes`` + ``blosc``
    codec chain, ``/``-separated ``c/<t>/0/0/0`` chunk keys — exactly what
    :mod:`biohub_tracking.ngff` targets.
    """
    array_dir.mkdir(parents=True, exist_ok=True)
    chunk_shape = [1, *data.shape[1:]]
    meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(data.shape),
        "data_type": str(data.dtype),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk_shape}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 1}},
        ],
    }
    (array_dir / "zarr.json").write_text(json.dumps(meta))
    for t in range(data.shape[0]):
        chunk_dir = array_dir / "c" / str(t) / "0" / "0"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        raw = np.ascontiguousarray(data[t]).tobytes()
        (chunk_dir / "0").write_bytes(_blosc_compress(raw))
    return array_dir


def test_ngff_reads_bare_array(tmp_path):
    """The reader reproduces every timepoint of a bare zarr-v3 array exactly."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, size=(3, 6, 16, 24), dtype=np.uint16)
    zpath = _write_competition_zarr(tmp_path / "vol.zarr", data)

    ng = open_ome_ngff_array(zpath)
    assert ng.shape == data.shape
    for t in range(data.shape[0]):
        assert np.array_equal(ng[t], data[t])


def test_ngff_reads_multiscale_level0(tmp_path):
    """The reader finds level ``0`` of an OME-multiscale ``*.zarr`` group."""
    data = np.arange(2 * 4 * 8 * 8, dtype=np.uint16).reshape(2, 4, 8, 8)
    _write_competition_zarr(tmp_path / "vol.zarr" / "0", data)

    ng = open_ome_ngff_array(tmp_path / "vol.zarr")
    assert ng.shape == data.shape
    for t in range(data.shape[0]):
        assert np.array_equal(ng[t], data[t])


def test_pipeline_falls_back_to_ngff_when_zarr_absent(tmp_path, monkeypatch):
    """``_open_image_array`` uses the ngff reader when ``import zarr`` fails."""
    data = np.arange(2 * 4 * 8 * 8, dtype=np.uint16).reshape(2, 4, 8, 8)
    zpath = _write_competition_zarr(tmp_path / "vol.zarr", data)

    import builtins

    real_import = builtins.__import__

    def _no_zarr(name, *args, **kwargs):
        if name == "zarr":
            raise ImportError("simulated: kernel image without zarr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_zarr)
    arr = pipeline_mod._open_image_array(zpath)
    assert arr.shape == data.shape
    assert np.array_equal(arr[0], data[0])
