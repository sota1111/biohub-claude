"""Generate the Kaggle submission CSV from the reigning champion (SOT-1984).

Runs the champion detection+linking pipeline over every ``*.zarr`` OME-NGFF test
video and writes a single competition submission CSV. This is the reproducible,
cwd-independent entry point the submission stage drives::

    python submit/make_submission.py --test-dir data/test --out submission.csv

It is a thin, exec-safe wrapper over :func:`biohub_tracking.build_submission`:
the champion params are resolved via :mod:`biohub_tracking.champion` (which falls
back to an embedded config when no filesystem is available), so this works from
any working directory. Deterministic — identical inputs yield an identical CSV.

Before submitting, run the exec-compat gate (``submit/exec_compat_gate.py``); it
is also wired into the test suite (``tests/test_exec_compat.py``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from biohub_tracking.build_submission import build_submission
from biohub_tracking.io import load_submission_csv


def make_submission(test_dir: Path | str, out_csv: Path | str, max_t: int | None = None) -> Path:
    """Build the champion submission CSV and verify it re-parses cleanly."""
    out = build_submission(test_dir, out_csv, max_t=max_t)
    # Round-trip check: the file we just wrote must load back as valid graphs.
    graphs = load_submission_csv(out)
    total_nodes = sum(g.num_nodes for g in graphs.values())
    total_edges = sum(g.num_edges for g in graphs.values())
    print(
        f"submission: {len(graphs)} datasets, "
        f"{total_nodes} nodes, {total_edges} edges -> {out}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"),
                        help="Dir of *.zarr test videos (default: data/test).")
    parser.add_argument("--out", type=Path, default=Path("submission.csv"),
                        help="Output submission CSV path (default: submission.csv).")
    parser.add_argument("--max-t", type=int, default=None, help="Cap timepoints (debug).")
    args = parser.parse_args()
    make_submission(args.test_dir, args.out, max_t=args.max_t)


if __name__ == "__main__":
    main()
