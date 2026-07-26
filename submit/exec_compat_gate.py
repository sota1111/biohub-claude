"""Kaggle exec-compatibility gate for the biohub-claude submission (SOT-1984).

The competition scores a flat submission CSV, but the code that *produces* it
must survive the constraints a Kaggle submission runtime imposes — the same
constraints the ``kaggle-exec-runtime-gate`` lesson records for the other
targets:

* **no ``__file__``** — code may be run via ``exec()`` / pasted into a kernel
  cell, where the module-level ``__file__`` name is not bound;
* **undefined cwd** — the process may start in an arbitrary directory, so
  nothing may depend on being launched from the repo root.

This gate exercises the champion's critical path under exactly those conditions
and asserts the output matches the SOT-1982 submission schema. It is deliberately
tiny and filesystem-free (a synthetic in-memory volume) so it runs in well under
a second, and is imported by ``tests/test_exec_compat.py`` so the quality gate
enforces it.

Run standalone::

    python submit/exec_compat_gate.py    # prints OK / raises on failure
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from biohub_tracking.champion import (
    EMBEDDED_CHAMPION_CONFIG,
    champion_params,
    load_champion_config,
)
from biohub_tracking.io import CSV_COLUMNS

# The champion code path, as a source string executed in a namespace WITHOUT a
# bound ``__file__`` — this is the "run under exec() / kernel cell" simulation.
_PIPELINE_SRC = """
import numpy as np
from biohub_tracking.champion import champion_params
from biohub_tracking.detect import detect_volume_series
from biohub_tracking.link import link_centroids
from biohub_tracking.io import write_submission_csv, load_submission_csv

assert "__file__" not in globals(), "exec namespace unexpectedly has __file__"

detect_params, link_params, scale = champion_params()

# Synthetic 2-timepoint (T, Z, Y, X) volume with two bright nuclei that persist
# across frames, so detection finds cells and linking makes edges.
rng = np.zeros((2, 8, 32, 32), dtype=np.float32)
for t in range(2):
    for (z, y, x) in [(4, 8, 8), (4, 24, 24)]:
        rng[t, z, y, x] = 1000.0
rng += 1.0  # nonzero floor so the percentile threshold is well-defined

detections = detect_volume_series(rng, detect_params)
graph = link_centroids(detections, scale=scale, params=link_params)
assert graph.num_nodes > 0, "champion detected no cells in the smoke volume"

out = write_submission_csv({"smoke": graph}, out_csv)
reloaded = load_submission_csv(out)
result["num_nodes"] = graph.num_nodes
result["num_edges"] = graph.num_edges
result["datasets"] = sorted(reloaded)
result["csv_path"] = str(out)
"""


def _run_pipeline_without_dunder_file(out_csv: Path) -> dict:
    """Execute the champion pipeline in a namespace with no ``__file__`` bound."""
    namespace: dict = {"out_csv": out_csv, "result": {}}
    # Note: exec with a fresh dict as globals => no __file__ key present.
    exec(compile(_PIPELINE_SRC, "<exec-compat-smoke>", "exec"), namespace)
    return namespace["result"]


def _assert_schema(csv_path: Path) -> None:
    """The CSV header + every row must match the SOT-1982 submission schema."""
    import csv as _csv

    with open(csv_path, newline="") as fh:
        reader = _csv.reader(fh)
        header = next(reader)
        assert tuple(header) == CSV_COLUMNS, f"bad header: {header!r}"
        int_cols = {"id", "node_id", "t", "z", "y", "x", "source_id", "target_id"}
        col_idx = {name: i for i, name in enumerate(header)}
        n_rows = 0
        for row in reader:
            assert len(row) == len(CSV_COLUMNS), f"ragged row: {row!r}"
            rtype = row[col_idx["row_type"]]
            assert rtype in ("node", "edge"), f"bad row_type: {rtype!r}"
            for c in int_cols:
                int(row[col_idx[c]])  # raises if not an integer
            n_rows += 1
        assert n_rows > 0, "submission CSV has no data rows"


def run_gate() -> dict:
    """Run the full exec-compat gate; return a summary dict or raise on failure."""
    # 1) Embedded fallback must byte-for-byte match the checked-in champion, so
    #    a filesystem-free kernel run uses the real champion params.
    on_disk = load_champion_config()
    assert on_disk == EMBEDDED_CHAMPION_CONFIG, (
        "EMBEDDED_CHAMPION_CONFIG drifted from champion/config.json — "
        "update biohub_tracking.champion.EMBEDDED_CHAMPION_CONFIG"
    )
    # champion_params must build from the embedded config with zero filesystem.
    champion_params(EMBEDDED_CHAMPION_CONFIG)

    # 2) Run the pipeline under undefined-cwd + no-__file__ and validate output.
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # simulate an arbitrary working directory
        try:
            out_csv = Path(tmp) / "sub" / "submission.csv"
            summary = _run_pipeline_without_dunder_file(out_csv)
            _assert_schema(Path(summary["csv_path"]))
        finally:
            os.chdir(original_cwd)
    return summary


def main() -> None:
    summary = run_gate()
    print(
        "exec-compat gate OK: "
        f"nodes={summary['num_nodes']} edges={summary['num_edges']} "
        f"datasets={summary['datasets']}"
    )


if __name__ == "__main__":
    main()
