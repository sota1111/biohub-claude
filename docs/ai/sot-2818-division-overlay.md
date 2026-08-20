# SOT-2818 — Non-destructive division-event overlay (default-OFF; not promoted)

**Cycle:** 2nd-series (biohub-claude, parent SOT-2815) · **Result:** rejected / do-not-promote ·
**Champion:** unchanged (`detect-link-dog-v4-shorttrack`, CV 0.6649 byte-for-byte)

## Axis

Recover the official metric's forfeited `0.1 · division_jaccard` term. The champion runs
`link.allow_division=false`, so it predicts zero forks and scores division Jaccard 0.0. Turning
division ON *inside* the linker (SOT-2762) was rejected: enabling `allow_division` re-runs the LAP
assignment, reassigns leftover detections and re-picks the primary daughter, spraying fork FPs on the
dense `6bba` families and losing edge TP across all 16 variants.

This axis is a **different mechanism**: a non-destructive post-processing overlay
(`src/biohub_tracking/division_overlay.py`) applied to the champion's already-linked, already-pruned
graph. It never touches the linking assignment — it only *re-attaches the second daughter the
one-to-one linker dropped*. At a real division the parent at `t` splits into two daughters at `t+1`;
the champion's optimal 1-to-1 assignment attaches the parent to only one of them, leaving the other as
a fresh **head**. The overlay adds the single edge `parent → nearby-persistent-head`, turning the
parent into a `1 → 2` fork. No node is added/removed, no existing edge is moved, so **OFF ⇒ champion
byte-for-byte** (default; `division_overlay=None`).

High-precision gates (each added edge is scored by the *edge* metric too, so a wrong second daughter is
both a division FP and an edge FP): `max_distance` (parent→D2 ≤ N µm), `sibling_ratio`
(`d2 ≤ ratio·d1` vs the primary daughter), `min_daughter_len` (D2 must persist ≥ k frames),
`require_parent_track` (parent must have a predecessor).

Ported concept: the public frontier lineage tracker's division handling —
<https://www.kaggle.com/code/prvsiyan/biohub-clean-public-frontier-lineage-tracker> (the same notebook
the SOT-2369 short-track filter came from). Metric rules:
<https://github.com/royerlab/kaggle-cell-tracking-competition>.

## Opportunity size (measured on real GT)

GT divisions per CV holdout family: `44b6_0113de3b=0`, `44b6_0b24845f=0`, `6bba_05b6850b=0`,
`6bba_05db0fb1=3` — **3 total, all in one video.** Division Jaccard is micro-averaged, so the absolute
ceiling is `+0.1` (all 3 recovered, zero FP). Any fork in the three zero-division families is a pure
division FP.

## Screen (SOT-2817 re-anchored full-metric CV; frozen detection, same-seed A/B)

`experiments/sot2818-division-overlay/screen_overlay.py` — 36 variants over
`max_distance∈{7,5,3}`, `sibling_ratio∈{0,2,1.5,1}`, `min_daughter_len∈{2,3,4}`. Baseline champion
reproduced **0.6649** exactly with the new code path.

| variant | division_jaccard | Δscore | per-family edge non-regression |
| --- | --- | --- | --- |
| best "promotable" `md=3, ratio=0, mdl=2` | **0.0** | +0.0004 | ✅ (but see below) |
| max division recovery `md=5, ratio=1.5, mdl=2` | 0.0909 (1 TP) | +0.0084 | ❌ regresses `6bba_05b6850b` 0.5700→0.5685 |
| strict symmetric `md=7, ratio=1.0` | 0.0 | +0.0000 | ✅ (fires nothing) |

## Decision — do NOT promote (keep default-OFF)

1. **The axis objective is unachievable under the non-regression gate.** No per-family-edge-non-regressing
   variant recovers *any* division TP — every "promotable" variant has `division_jaccard=0.0`.
2. **Recovering the division term costs edge regression.** Every variant that recovers a real division
   TP (up to `div_j=0.0909`, +0.0084 combined) regresses the sibling dense family `6bba_05b6850b`
   (adj 0.5700→0.5685): that family has **0** GT divisions, so each added second-daughter fork there is
   a pure division/edge FP. This is the **same failure mode SOT-2762 was rejected for**, reproduced
   through the non-destructive mechanism — new evidence that the barrier is the sparse, single-video
   division signal on this holdout, not the (previously suspected) LAP reassignment.
3. **The only non-regressing gain is an artifact, and family-mix-sensitive.** The `md=3` variants'
   +0.0004 comes purely from incidental **edge**-TP recovery (`6bba_05b6850b` edge TP 651→652) with
   `division_jaccard=0.0` (division term still forfeited, plus 1 division FP). The gain sits entirely on
   the dominant lineage: `representativeness_report` reports `family_mix_sensitive=True`
   (micro↔lineage-macro gap 0.0565 > 0.05 tol), exactly the dominant-lineage hair-gain the SOT-2817
   guard flags as **insufficient** evidence. Promoting a +0.06% family-mix-sensitive micro delta risks
   repeating the SOT-2816 CV↑/LB↓ hazard.

**Outcome:** the overlay is implemented and unit-tested but stays **default-OFF**;
`champion/config.json`/`registry.json` are unchanged; the champion CV is 0.6649 byte-for-byte
(`--check-champion` reproduces). No Kaggle submission. Full artefacts:
`experiments/sot2818-division-overlay/{screen_overlay,confirm_overlay}.json`.
