"""SOT-2930 A/B: adopt the public classical baseline *foundation* WHOLESALE (leak-free CV).

Cycle-6 explore direction A' (role A' hedge). Our champion
(``detect-link-dog-v4-shorttrack-motion-gain1``, leak-free CV micro_adj **0.6760**,
public 0.626) was built by *incremental single-lever accretion* on top of a classical
DoG-detect / nearest-neighbour-link substrate and is stuck at a local optimum on the
density-mix wall (7+ linking axes rejected: SOT-2864 promoted, then SOT-2895/2898/2899/
2910/2911/2918/2920/2922/2923/2931 all rejected/inconclusive).

This axis does NOT tune the champion. It adopts an **independently-constructed public
classical baseline foundation WHOLESALE** and measures it, as an independent candidate,
on the same leak-free 4-family LOFO CV. The champion is kept as hedge (byte-frozen).

Adopted source (offline / numpy-scipy / no-weights / deterministic — the portable part):

  * ``xiaoleilian/biohub-cell-tracking-classical-baseline`` (public LB ~0.720)
    https://www.kaggle.com/code/xiaoleilian/biohub-cell-tracking-classical-baseline
    — DoG blob detection + **memoryless two-pass Hungarian linking** ("v4: tight gate
    first, then full gate for the leftovers", tight 7 um / full ~11 um) with division.
  * ``kaiwalyaatulraut/biohub-cell-tracking-solution`` (classical solution NB) —
    corroborating classical DoG + greedy NN family.

Portability judgement (rogii discipline, recorded in the ledger):
  * PORTABLE (adopted wholesale here): the baseline's *structural linking foundation*
    — memoryless two-pass tight-then-full Hungarian assignment + its division-on /
    no-aggressive-short-track-prune envelope. This is the basin the baseline lives in,
    structurally DISTINCT from our champion's ARGUS global-motion-model LAP linker.
  * SHARED SUBSTRATE: DoG blob detection. Our champion detection IS itself the ported
    classical-baseline-family DoG detector (``detect-link-dog-v4``); the baseline's
    differentiator vs our champion is the *linker*, not the detector. We therefore hold
    detection at the shared classical DoG operating point (which also caches detection
    once per family, so only the linking foundation varies — a clean linking-only
    ablation). The private notebook's exact detection thresholds are not byte-recoverable
    offline; this is a faithful reconstruction from the documented design + the repo's
    already-ported baseline linker knobs (``link_two_pass``/``link_full_distance``,
    SOT-2899), NOT a byte-copy of the private kernel source. DISCLOSED.
  * NON-PORTABLE (excluded): none for the *classical* baseline (no GPU weights). The
    frontier 0.89+ learned UNet+ILP lineages are excluded by scope (offline/no-weights).

rogii guard: the adopted baseline's high PUBLIC (0.720) is NOT private evidence and is
NOT used to justify promotion. Promotion requires the two-signal gate: leak-free CV
micro_adj UP beyond the noise band AND 4/4 per-dataset non-regression (adjusted). If the
wholesale foundation does not clear that gate, it is NOT promoted; the champion stays the
live config (byte-frozen) and — only if the foundation is genuinely CV-competitive and
structurally independent — is retained as a final-slot hedge. No Kaggle submission.

Writes ``experiments/sot2930/ab_public_classical_foundation.json``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import load_champion_config
from biohub_tracking.eval.cv import cv_result_to_dict, evaluate_cv

REPO = Path(__file__).resolve().parents[2]
CHAMPION_REFERENCE_MICRO_ADJ = 0.6760  # live champion (SOT-2909), registry.json
NOISE_BAND = 0.005


def _sha256(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _foundation_variants(champ: dict) -> dict[str, dict]:
    """The wholesale public-classical-baseline linking foundation, in faithful variants.

    All share the champion's classical DoG detection substrate (disclosed above); the
    LINK block is replaced wholesale by the baseline's memoryless two-pass foundation.
    The champion's signature lever (``motion_model_link``, the ARGUS global motion field)
    is turned OFF — this is the baseline's own basin, not a champion micro-diff.
    """
    variants: dict[str, dict] = {}

    # W1 — baseline linker core: memoryless two-pass tight(7)->full(11) Hungarian, the
    # champion's division/prune envelope (allow_division=false, min_track_length=4). This
    # is the baseline linker adopted with our champion's conservative post-envelope.
    w1 = copy.deepcopy(champ)
    w1["name"] = "public-classical-foundation-w1-twopass-core"
    w1["link"] = {
        "max_distance": 7.0,
        "allow_division": False,
        "division_distance": 7.0,
        "min_track_length": 4,
        "link_two_pass": True,
        "link_full_distance": 11.0,
    }
    variants["w1_twopass_core"] = w1

    # W2 — baseline wholesale with division ON (the baseline handles divisions; the metric
    # carries a 0.1*division term the champion forgoes with allow_division=false). Second
    # daughter attached within the full gate.
    w2 = copy.deepcopy(champ)
    w2["name"] = "public-classical-foundation-w2-twopass-division"
    w2["link"] = {
        "max_distance": 7.0,
        "allow_division": True,
        "division_distance": 11.0,
        "min_track_length": 4,
        "link_two_pass": True,
        "link_full_distance": 11.0,
    }
    variants["w2_twopass_division"] = w2

    # W3 — baseline wholesale, no aggressive short-track pruning (min_track_length=1): the
    # baseline keeps short real fragments rather than pruning to >=4, a distinct
    # recall/precision envelope from our champion.
    w3 = copy.deepcopy(champ)
    w3["name"] = "public-classical-foundation-w3-twopass-division-nopruning"
    w3["link"] = {
        "max_distance": 7.0,
        "allow_division": True,
        "division_distance": 11.0,
        "min_track_length": 1,
        "link_two_pass": True,
        "link_full_distance": 11.0,
    }
    variants["w3_twopass_division_nopruning"] = w3

    return variants


def _summ(cfg: dict) -> dict:
    r = evaluate_cv(config=cfg)
    d = cv_result_to_dict(r)
    per = {
        row["name"]: row["adjusted_edge_jaccard"] for row in d["per_dataset"]
    }
    return {
        "micro_adj": d["micro_adj_edge_jaccard"],
        "micro_raw": d["micro_edge_jaccard"],
        "macro_adj": d["macro_adj_edge_jaccard"],
        "lineage_macro_adj": d["lineage_macro_adj"],
        "division_jaccard": d["division_jaccard"],
        "score": d["score"],
        "per_dataset_adj": per,
        "_result": r,
    }


def main() -> None:
    t0 = time.time()
    champ = load_champion_config(REPO / "champion" / "config.json")
    champ_sha = _sha256(champ)

    print("evaluating champion (anchor) ...", flush=True)
    champ_s = _summ(champ)
    champ_per = champ_s["per_dataset_adj"]
    print(
        f"  champion micro_adj={champ_s['micro_adj']:.4f} "
        f"(registry ref {CHAMPION_REFERENCE_MICRO_ADJ})",
        flush=True,
    )

    variants = _foundation_variants(champ)
    results: dict[str, dict] = {}
    for key, cfg in variants.items():
        print(f"evaluating {key} ...", flush=True)
        s = _summ(cfg)
        r = s.pop("_result")
        no_reg = r.no_regression_vs(champ_per)
        delta = s["micro_adj"] - champ_s["micro_adj"]
        per_delta = {
            k: round(s["per_dataset_adj"][k] - champ_per.get(k, float("nan")), 4)
            for k in s["per_dataset_adj"]
        }
        beats = delta > NOISE_BAND and no_reg
        results[key] = {
            **s,
            "delta_micro_adj": round(delta, 4),
            "per_dataset_delta_adj": per_delta,
            "no_regression_vs_champion": no_reg,
            "beats_champion_two_signal_gate": beats,
        }
        print(
            f"  {key}: micro_adj={s['micro_adj']:.4f} d={delta:+.4f} "
            f"no_reg={no_reg} div_jac={s['division_jaccard']} gate_pass={beats}",
            flush=True,
        )

    champ_s.pop("_result", None)
    best_key = max(results, key=lambda k: results[k]["micro_adj"])
    any_promote = any(v["beats_champion_two_signal_gate"] for v in results.values())

    payload = {
        "issue": "SOT-2930",
        "axis": "cycle-6 direction A' — wholesale public classical baseline foundation",
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "champion": {
            "name": champ["name"],
            "config_sha256": champ_sha,
            "registry_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
            **champ_s,
        },
        "sources": [
            {
                "lineage": "xiaoleilian/biohub-cell-tracking-classical-baseline",
                "url": "https://www.kaggle.com/code/xiaoleilian/biohub-cell-tracking-classical-baseline",
                "public_lb": 0.720,
                "portable": "classical DoG detect + memoryless two-pass tight/full Hungarian link + division (offline/numpy/no-weights)",
            },
            {
                "lineage": "kaiwalyaatulraut/biohub-cell-tracking-solution",
                "url": "https://www.kaggle.com/code/kaiwalyaatulraut/biohub-cell-tracking-solution",
                "public_lb": None,
                "portable": "corroborating classical DoG + greedy NN family",
            },
        ],
        "noise_band": NOISE_BAND,
        "variants": results,
        "best_variant": best_key,
        "best_variant_micro_adj": results[best_key]["micro_adj"],
        "any_variant_promotes": any_promote,
        "rogii_note": (
            "adopted baseline public 0.720 > champion public 0.626, but public is NOT "
            "private evidence; promotion decided ONLY by leak-free CV two-signal gate "
            "(micro_adj up + 4/4 non-regression). Champion byte-frozen as hedge."
        ),
        "kaggle_submission": False,
        "elapsed_s": round(time.time() - t0, 1),
    }

    out = REPO / "experiments" / "sot2930" / "ab_public_classical_foundation.json"
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"VERDICT any_promote={any_promote} best={best_key} "
        f"best_micro_adj={results[best_key]['micro_adj']:.4f} "
        f"champion={champ_s['micro_adj']:.4f} champion_byte_frozen=True",
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
