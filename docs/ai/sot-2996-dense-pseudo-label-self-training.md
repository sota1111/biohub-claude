# SOT-2996 — Semi-supervised dense pseudo-label self-training

**Result: NON-PROMOTED (rejected for promotion; mechanism partially validated, with
evidence). Champion byte-frozen (CV micro_adj `0.6760`, threshold=0 reproduces it
byte-for-byte in both arms). Kaggle NOT submitted. Module ships default-off.**

Cycle-7 (SOT-2992) direction 4 — escalation-ladder **step-4 problem reformulation**.
Attacks the *root cause* of the density-mix wall (sparse annotation) by densifying the
**teacher signal**, not by tuning an operating point (which every prior op-point axis
failed to do). Runnable standalone: the teacher bootstraps from the classical
champion detector, so it does **not** hard-block on the SOT-2993 learned detector.

## Axis

The learned detection scorer (SOT-2828) was REJECTED because the sparse GT poisons its
labels: the competition annotates only *one* lineage (~52 nodes) out of ~25 000 true
cells per volume, so `label_candidates` labels every un-annotated real cell as a
(down-weighted) **negative** — a Positive-Unlabeled set that teaches the scorer to
rank real cells *low*. On the SOT-2828 LOFO screen the held-out positive-minus-negative
probability gap was **negative on 3/4 families** (a real cell scored below noise).

This axis densifies the supervision instead. The classical champion detector's own
**high-confidence** (`resp_z ≥ pooled-train p90`, leak-free) and
**temporally-consistent** (champion-linked weakly-connected track length ≥ 3, a GT-free
motion-coherence signal) un-annotated detections are promoted to **dense
pseudo-positives**; candidates that fail the gate become **confident negatives**
(full weight, no longer down-weighted as "maybe real"). The scorer is then LOFO-trained
on the densified labels (arm B) and compared same-seed against the SOT-2828 sparse-GT PU
labeling (arm A), both vs the byte-frozen `detect-link-dog-v4-shorttrack` champion.

`src/biohub_tracking/pseudo_label.py` — pure numpy/scipy, no torch, exec-compat,
**default-off** (the champion pipeline never imports it; the byte-frozen champion is
unchanged). It produces a drop-in `(labels, weights)` replacement for
`detect_scorer.label_candidates` in a scorer-training screen; it alters no promoted
config.

## Evaluation

Leak-free leave-one-family-out (LOFO), identical to SOT-2828: each of the 4 CV families
is scored by a scorer trained only on the other three, aggregated through the SOT-2817
re-anchored full-metric CV (byte-comparable to the registry champion). The confidence
gate threshold is computed **per fold on the pooled TRAINING families** (held-out
excluded → leak-free). `threshold=0.0` keeps every candidate ⇒ reproduces the champion
byte-for-byte (sanity: **True**, both arms). Detection + features + champion track
length are cached once; only the labeling scheme and the keep-mask threshold change.
No Kaggle submission. Screen: `experiments/sot2996/screen_pseudo_label.py`; results
`experiments/sot2996/screen_pseudo_label.json`.

## Findings

**1. The PU-poisoning is genuinely repaired at the ranking level (hypothesis
confirmed).** Mean held-out pos−neg prob gap flips from **arm A = −0.0274** (SOT-2828
failure reproduced: real cells below noise) to **arm B = +0.0867** (real cells now rank
above noise). The repair is largest exactly on the **sparse 6bba regime**
(`6bba_05b6850b`): **A = −0.1045 → B = +0.3341**. Dense pseudo-labeling does attack the
root cause SOT-2828 identified.

| held-out family | regime | arm A prob_gap | arm B prob_gap |
| --- | --- | --- | --- |
| 44b6_0113de3b | dense | −0.0300 | +0.0109 |
| 44b6_0b24845f | dense | −0.0105 | +0.0066 |
| 6bba_05b6850b | **sparse** | **−0.1045** | **+0.3341** |
| 6bba_05db0fb1 | dense | +0.0353 | −0.0047 |
| **mean** | | **−0.0274** | **+0.0867** |

**2. But it does NOT translate into a promotable CV operating point:
`n_promotable = 0` in both arms.** Only `threshold=0` (keep-all = champion) holds
non-regression, and it does not improve the score. Any `threshold > 0` used as a
candidate filter drops both TP and nodes and tanks `micro_adj` (arm B best filtered
`micro_adj = 0.2992` @ t=0.4, Δscore −0.377, `dNodes` −108k..−129k). **The density-mix
wall persists at the filtering stage**: the champion's keep-all feasible set is already
saturated near GT, so a better *ranking* still can't be converted into a better
*selection* — the third+ confirmation of the SOT-2841/2870/2994 "discrimination ≠ CV
gain" lesson, now at the labeling layer.

**3. The confidence gate mis-calibrates across families (density-mix in the gate).**
`gate_recall_on_gt` (fraction of known-real GT cells the gate accepts) is **0.94 on
`6bba_05b6850b`** but **~0.0–0.06 on the low-response 44b6 families** — the pooled-train
p90 `resp_z` threshold (up to 102–120 on 44b6 folds) is dominated by 6bba's
high-response blobs and rejects nearly all of 44b6's genuine (lower-response) cells.
`resp_z` is not on a common scale across families, so a single global percentile gate
densifies unevenly. Per-family / rank-normalised confidence calibration is the obvious
next lever if this axis is revisited.

**4. Consistency separates signal from noise as designed.** mean track length of
GT-positive vs gate-rejected candidates: `6bba_05b6850b` 24.3 vs 6.5, `44b6_0113de3b`
41.6 vs 27.9 — temporal coherence is a real FP-suppression signal. (The
`pseudo_fp_upper_bound = 1.0` diagnostic is a *deliberate* massive over-count — the
sparse GT annotates only one lineage, so every pseudo-positive at an annotated timepoint
that isn't THAT lineage counts as "FP"; it is reported for transparency, never used as a
gate.)

## Verdict

**Rejected for promotion; mechanism partially validated.** Dense pseudo-label
self-training repairs the SOT-2828 label-poisoning at the ranking level (prob_gap flips
positive, strongly on the sparse 6bba regime), confirming that sparse annotation — not
the scorer — was the SOT-2828 failure cause. But the repaired ranking does not clear the
non-regression gate: the density-mix wall re-appears (a) at candidate selection (any
filter loses TP) and (b) in the cross-family scale of the confidence gate. The champion
is byte-frozen and untouched (module default-off, kept as infrastructure alongside the
other default-off learned modules); champion hedge preserved. No revert of any promoted
artifact was needed. No Kaggle submission.

**Headroom** (consistent with SOT-2993/2994): the wall is in the detection *substrate*
and the *selection* step, not in ranking quality. Next levers if revisited:
per-family/rank-normalised confidence calibration; using pseudo-labels to retrain the
*detector* (SOT-2993) rather than a candidate-filter scorer; or a per-regime operating
point (which requires the SOT-2921 covariate to be usable at inference, still open).

## Bug fixed en route

`track_length_per_candidate` crashed the screen with
`track_len rows (34818) != candidate rows (42080)`: it linked with the **champion**
params, whose `min_track_length = 4` prunes short tracks (and re-numbers node ids), so
the pruned graph had fewer nodes than candidates and node-id ≠ row-index. Fixed by
disabling only the two node-*set* mutators (`min_track_length → 1`,
`gap_recover → False`) via `dataclasses.replace`, keeping every association knob at the
champion's value — so the graph now has exactly one node per candidate in `row_t` order
and singletons are retained as length-1 tracks (which is precisely what the consistency
gate must reject). Pinned by
`tests/test_pseudo_label.py::test_track_length_one_row_per_candidate_when_params_prune`.

## Artifacts

- `src/biohub_tracking/pseudo_label.py` — dense pseudo-label densifier (default-off).
- `tests/test_pseudo_label.py` — 5 data-free tests (incl. the prune regression).
- `experiments/sot2996/screen_pseudo_label.py` — leak-free LOFO A/B screen.
- `experiments/sot2996/screen_pseudo_label.json` — full results.
