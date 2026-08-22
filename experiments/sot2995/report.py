"""SOT-2995 step 3 (repo .venv): assemble the oracle-fidelity + transfer-trust report.

Pure aggregation over the JSON that ``predict.py`` (clean-room) and
``score_official.py`` (official) wrote. Produces:

* **Divergence**: per-family official−clean count deltas + the aggregate CV each
  scorer reports (criterion: divergence vs the custom micro_adj quantified).
* **Transfer-trust**: the official-scorer CV micro_adj re-scored for each known
  submission, and its Spearman rank-correlation / top-match vs the same-metric
  public anchors (criterion: known 0.509 / 0.626 re-scored, order agreement
  recorded).

Writes ``docs/ai/sot-2995-oracle-fidelity.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from biohub_tracking.eval.cv import FamilyResult, aggregate
from biohub_tracking.eval.transfer import spearman

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path("/tmp/sot2995")

# Same-metric public anchors (pre division-exploit-patch footing): the two known
# submissions this issue names, 0.509 (v1) and 0.626 (champion motion-link).
PUBLIC_ANCHORS = {"v1": 0.509, "champion": 0.626}


def _rows_to_family_results(rows: list[dict]) -> list[FamilyResult]:
    return [
        FamilyResult(
            name=r["name"],
            lineage=r["lineage"],
            edge_tp=r["edge_tp"],
            edge_fp=r["edge_fp"],
            edge_fn=r["edge_fn"],
            edge_jaccard=r["edge_jaccard"],
            adj_edge_jaccard=r["adj_edge_jaccard"],
            division_tp=r["division_tp"],
            division_fp=r["division_fp"],
            division_fn=r["division_fn"],
            num_pred_nodes=r["num_pred_nodes"],
            n_true=r["n_true"] if r["n_true"] is not None else float("nan"),
            weight=r["weight"],
        )
        for r in rows
    ]


def _round(v, n=4):
    return None if (isinstance(v, float) and math.isnan(v)) else round(v, n)


def main() -> int:
    labels = [d.name for d in sorted(OUT_ROOT.iterdir()) if d.is_dir()]

    per_label = {}
    for label in labels:
        clean_path = OUT_ROOT / label / "clean_scores.json"
        official_path = OUT_ROOT / label / "official_scores.json"
        if not (clean_path.exists() and official_path.exists()):
            continue
        clean = json.loads(clean_path.read_text())
        official = json.loads(official_path.read_text())

        clean_cv = aggregate(_rows_to_family_results(clean["rows"]))
        official_cv = aggregate(_rows_to_family_results(official["rows"]))

        per_label[label] = {
            "clean_micro_adj": _round(clean_cv.micro_adj_edge_jaccard),
            "official_micro_adj": _round(official_cv.micro_adj_edge_jaccard),
            "clean_score": _round(clean_cv.score),
            "official_score": _round(official_cv.score),
            "micro_adj_divergence": _round(
                official_cv.micro_adj_edge_jaccard - clean_cv.micro_adj_edge_jaccard, 6
            ),
            "public_lb": PUBLIC_ANCHORS.get(label),
            "per_family_divergence": official["divergence"],
        }

    # Transfer-trust: official micro_adj vs same-metric public over anchored labels.
    anchored = [
        (lbl, per_label[lbl])
        for lbl in per_label
        if per_label[lbl]["public_lb"] is not None
    ]
    off_vals = [d["official_micro_adj"] for _, d in anchored]
    clean_vals = [d["clean_micro_adj"] for _, d in anchored]
    pub_vals = [d["public_lb"] for _, d in anchored]

    def _order(vals):
        rho = spearman(vals, pub_vals)
        top_idx = max(range(len(vals)), key=lambda i: vals[i]) if vals else None
        pub_top_idx = (
            max(range(len(pub_vals)), key=lambda i: pub_vals[i]) if pub_vals else None
        )
        return {
            "n_anchored": len(vals),
            "spearman_vs_public": (None if math.isnan(rho) else round(rho, 4)),
            "cv_top_matches_public_top": (
                None if top_idx is None else top_idx == pub_top_idx
            ),
            "pairs": [
                {"label": anchored[i][0], "cv": vals[i], "public_lb": pub_vals[i]}
                for i in range(len(vals))
            ],
        }

    # Max per-family absolute count divergence across everything scored.
    max_abs = 0
    n_families = 0
    for d in per_label.values():
        for fam in d["per_family_divergence"]:
            n_families += 1
            for v in fam["delta"].values():
                max_abs = max(max_abs, abs(v))

    report = {
        "issue": "SOT-2995",
        "title": "Official evaluate.py ported to the CV scorer + transfer-trust re-anchor",
        "official_scorer": "royerlab/kaggle-cell-tracking-competition tracking_cellmot.metrics (tracksdata+polars)",
        "adjustment_alpha": 0.1,
        "division_weight": 0.1,
        "divergence": {
            "families_scored": n_families,
            "max_abs_count_delta": max_abs,
            "clean_room_is_faithful": max_abs == 0,
            "per_label": {
                lbl: {
                    "clean_micro_adj": d["clean_micro_adj"],
                    "official_micro_adj": d["official_micro_adj"],
                    "micro_adj_divergence": d["micro_adj_divergence"],
                    "clean_score": d["clean_score"],
                    "official_score": d["official_score"],
                    "public_lb": d["public_lb"],
                }
                for lbl, d in per_label.items()
            },
        },
        "transfer_trust": {
            "public_anchors": PUBLIC_ANCHORS,
            "official_scorer_order": _order(off_vals),
            "clean_room_order": _order(clean_vals),
        },
        "per_family_divergence": {
            lbl: d["per_family_divergence"] for lbl, d in per_label.items()
        },
    }

    out = REPO / "docs/ai/sot-2995-oracle-fidelity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["divergence"], indent=2))
    print(json.dumps(report["transfer_trust"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
