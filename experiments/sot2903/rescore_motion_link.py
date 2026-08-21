"""SOT-2903: re-score the SOT-2900 motion-link candidate under the TRUE metric.

True/official metric = micro-averaged ADJUSTED edge Jaccard + 0.1*div (cv.py
`score`). Emit micro_adj (true-metric proxy), micro_raw (guardrail: raw
matching), macro/lineage, and per-dataset non-regression vs the byte-frozen
champion, so the parent can judge promotion under the SOT-2903-repaired anchor.
"""
from __future__ import annotations
import json
from pathlib import Path
from biohub_tracking.eval.cv import evaluate_cv

CAND = "champion/candidates/sot2900-motion-model-link-gain2.json"

def summ(r):
    return {
        "micro_adj_edge_jaccard": round(r.micro_adj_edge_jaccard, 4),  # TRUE metric proxy
        "micro_raw_edge_jaccard": round(r.micro_edge_jaccard, 4),      # guardrail (raw matching)
        "macro_adj_edge_jaccard": round(r.macro_adj_edge_jaccard, 4),
        "lineage_macro_adj": round(r.lineage_macro_adj, 4),
        "score": round(r.score, 4),
        "per_dataset_adj": {fr.name: round(fr.adj_edge_jaccard, 4) for fr in r.per_dataset},
        "per_dataset_raw": {fr.name: round(fr.edge_jaccard, 4) for fr in r.per_dataset},
    }

champ = evaluate_cv()
cand = evaluate_cv(json.loads(Path(CAND).read_text()))
cs, ds = summ(champ), summ(cand)
adj_d = {k: round(ds["per_dataset_adj"][k]-cs["per_dataset_adj"][k], 4) for k in cs["per_dataset_adj"]}
raw_d = {k: round(ds["per_dataset_raw"][k]-cs["per_dataset_raw"][k], 4) for k in cs["per_dataset_raw"]}
out = {
    "candidate": CAND,
    "champion": cs,
    "candidate_cv": ds,
    "delta_micro_adj_TRUE_metric": round(ds["micro_adj_edge_jaccard"]-cs["micro_adj_edge_jaccard"], 4),
    "delta_micro_raw_guardrail": round(ds["micro_raw_edge_jaccard"]-cs["micro_raw_edge_jaccard"], 4),
    "per_dataset_adj_delta": adj_d,
    "per_dataset_raw_delta": raw_d,
    "improves_under_true_metric": ds["micro_adj_edge_jaccard"] > cs["micro_adj_edge_jaccard"],
    "adj_no_regression": all(v >= -1e-9 for v in adj_d.values()),
    "raw_no_regression_guardrail": all(v >= -1e-9 for v in raw_d.values()),
}
Path("experiments/sot2903/motion_link_rescore.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
