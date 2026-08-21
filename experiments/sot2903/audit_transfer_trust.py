"""SOT-2903 CV<->public transfer-trust audit over the public-known anchors.

Reuses the single leak-free harness lineage (biohub_tracking.eval.transfer.
HISTORICAL_LINEAGE) and the public LB anchors the SOT-2903 issue names
(48b1e=0.624 SOT-2369 champion; 01c2f3=0.557 SOT-2300 v3-adaptive; e445965=0.509
= the SAME champion re-scored post metric-patch). For every candidate CV
statistic it measures Spearman rank-correlation vs the public LB and whether the
CV's top config is the public top -- the concrete "is the CV a private/public
proxy" test. Pure: no competition data needed (rows are the frozen single-harness
re-score, guarded live by cv.py --check-champion == 0.6649).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from biohub_tracking.eval.transfer import (
    HISTORICAL_LINEAGE, ConfigCv, transfer_stats, order_consistency,
)

# The public-known anchors, on the SAME (pre-patch) metric footing where possible.
#   detect-link-v1             0.509  (2026-07-28, pre-patch)
#   detect-link-dog-v2         0.500  (near-tie, pre-patch)
#   detect-link-dog-v3-adaptive 0.557 (01c2f3, 2026-08-02 SOT-2300; ref/date-
#                                      attributed, byte-UNCONFIRMED per SOT-2902)
#   detect-link-dog-v4-shorttrack 0.624 (48b1e, 2026-08-03 SOT-2369 champion best)
V3_PUBLIC = 0.557

def lineage_with_v3() -> list[ConfigCv]:
    out = []
    for c in HISTORICAL_LINEAGE:
        if c.name == "detect-link-dog-v3-adaptive":
            out.append(c._replace(
                public_lb=V3_PUBLIC,
                public_note=("01c2f3 @v7 fallback identity, 2026-08-02 SOT-2300; "
                             "0.557 attributed by submission date/ref, byte-UNCONFIRMED "
                             "(SOT-2902: fallback identity hashes no CSV content)."),
            ))
        else:
            out.append(c)
    return out

STATS = ["micro_adj", "micro_raw", "macro_adj", "lineage_macro_adj", "lineage_macro_raw"]

def run(lineage: list[ConfigCv], label: str) -> dict:
    stats = [transfer_stats(c) for c in lineage]
    anchored = [s for s in stats if s.public_lb is not None]
    consistency = {st: order_consistency(stats, st) for st in STATS}
    # rank each statistic by (spearman desc, then top-match)
    ranked = sorted(
        STATS,
        key=lambda st: (
            -(consistency[st]["spearman_vs_public"] or -9),
            0 if consistency[st]["cv_top_matches_public_top"] else 1,
        ),
    )
    return {
        "label": label,
        "n_anchored": len(anchored),
        "anchors": [{"name": s.name, "public_lb": s.public_lb} for s in anchored],
        "order_consistency": consistency,
        "best_true_metric_proxy": ranked[0],
        "ranking_by_spearman": [
            {"stat": st, "spearman": consistency[st]["spearman_vs_public"],
             "cv_top_is_public_top": consistency[st]["cv_top_matches_public_top"]}
            for st in ranked
        ],
    }

def main() -> int:
    report = {
        "official_metric": {
            "adjusted_edge_jaccard": "max(0, J*(1 - 0.1*(N_pred-N_true)/N_true)), alpha=0.1",
            "score": "adjusted_edge_jaccard + 0.1*division_jaccard",
            "matching": "7um one-to-one optimal (min-cost) bipartite assignment",
            "aggregation": "micro-averaged (sum TP/FP/FN across the split, then Jaccard)",
            "source": "https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md",
        },
        "note": ("The official metric is MICRO-averaged ADJUSTED (penalty a=0.1) edge "
                 "Jaccard + 0.1*division. Among CV statistics, micro_adj is the "
                 "official-metric-faithful proxy; lineage_macro_raw (SOT-2894 re-anchor) "
                 "DROPS the penalty AND macro/lineage-parity-averages AND uses raw J, "
                 "i.e. diverges from the true metric on all three axes."),
        "with_v3_0557_anchor": run(lineage_with_v3(), "4 anchors incl. v3=0.557 (byte-unconfirmed)"),
        "without_v3_anchor": run(list(HISTORICAL_LINEAGE), "3 byte-anchored (v3 excluded, as SOT-2894)"),
    }
    out = Path("experiments/sot2903/transfer_trust_audit.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
