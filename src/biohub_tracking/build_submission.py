"""Build a submission CSV by running the champion pipeline over test videos.

For every ``*.zarr`` OME-NGFF video under ``--test-dir``, detect + link cells
with the reigning champion configuration (``champion/config.json``) and write all
datasets into a single submission CSV::

    python -m biohub_tracking.build_submission --test-dir data/test --out submission.csv

This is the reproducible entry point the submission stage (SOT-1984) drives; it
is deterministic, so the same inputs always yield the same CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .champion import champion_params, learned_detector_config, load_champion_config
from .io import write_submission_csv
from .pipeline import run_pipeline


def build_submission(
    test_dir: Path | str,
    out_csv: Path | str,
    config: dict | None = None,
    max_t: int | None = None,
    verbose: bool = True,
) -> Path:
    """Run the champion pipeline over every ``*.zarr`` in *test_dir*."""
    test_dir = Path(test_dir)
    if config is None:
        config = load_champion_config()
    detect_params, link_params, scale = champion_params(config)

    # Learned-detector receptacle (SOT-2847, default-off): a learned detector is
    # built only when the config carries an enabled ``learned_detector`` block. The
    # champion config has none, so this is ``None`` and the classical detect path
    # runs byte-for-byte unchanged.
    learned_detector = None
    ld_block = learned_detector_config(config)
    if ld_block is not None:
        from .learned_detect import build_learned_detector

        learned_detector = build_learned_detector(ld_block)

    videos = sorted(test_dir.glob("*.zarr"))
    if not videos:
        raise FileNotFoundError(f"no *.zarr videos found under {test_dir}")

    graphs = {}
    for video in videos:
        name = video.stem
        graph = run_pipeline(
            video,
            scale=scale,
            detect_params=detect_params,
            link_params=link_params,
            max_t=max_t,
            learned_detector=learned_detector,
        )
        graphs[name] = graph
        if verbose:
            print(f"{name}: nodes={graph.num_nodes} edges={graph.num_edges}")

    return write_submission_csv(graphs, out_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True, help="Dir of *.zarr videos.")
    parser.add_argument("--out", type=Path, required=True, help="Output submission CSV path.")
    parser.add_argument("--max-t", type=int, default=None, help="Cap timepoints (debug).")
    args = parser.parse_args()
    out = build_submission(args.test_dir, args.out, max_t=args.max_t)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
