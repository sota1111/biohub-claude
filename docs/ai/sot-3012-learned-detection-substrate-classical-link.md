# SOT-3012 — Isolating the learned-DETECTION lever: learned detection substrate × champion classical motion-link

**Cycle:** biohub-claude Kaggle 順位向上 第3次 (parent SOT-3010) · **Axis:** role A′ *detection-only*
hybrid (SOT-2992 申し送り軸#3, 方向2) · **Decision: REJECT (non-promotable) — but the detection lever is
confirmed and the remaining wall is localized to one family.**

## Question

SOT-3011 adopted the official royerlab released-weights pipeline **wholesale** (learned detection **and**
learned linking, greedy/ILP) and REJECTED 4/4: micro rose to 0.7705 (greedy) / 0.8081 (ILP) but **both**
sparse 44b6 families regressed (−0.024 … −0.111). That leaves the causal question open: is the huge dense-family
(6bba) gain a **detection-substrate** effect or a **linking** effect?

This issue answers it by swapping **only detection**: the learned detection **centres** are fed into the
**champion's classical ARGUS motion-model LAP linking**, held byte-fixed
(`motion_model_link=true, motion_smooth_sigma=15.0, motion_gain=1.0, min_track_length=4`). No learned linking is
reintroduced (SOT-2841/2870/2994 already saturated that axis 3×). Both arms are scored by the identical leak-free
harness (`biohub_tracking.eval.cv`), so the A/B isolates detection.

## Method

- **Learned arm:** `predict_video` (released `split_0` weights, TemporalUNet3D) → detection `coords[N,4]` in
  original-resolution voxel space → grouped to `{t:(M,3)}` → `link_centroids(..., scale=geff_scale, params=champion_link)`
  → `score_family`. The learned **edges** are discarded; only the detection substrate is used.
- **Champion arm:** `evaluate_cv()` (classical DoG+NMS detection + champion linking) → micro_adj **0.6760**.
- Same seed, same 4-family LOFO holdout, same re-anchored micro_adj. Detection sigmoid threshold swept
  `{0.5, 0.9, 0.99, 0.999}`.
- Infra reused from `experiments/sot3011/_ext` (clone + released weights); ran in `.venv` (torch 2.5.1+cu121, CUDA).

## Result — REJECT at every global threshold, failure collapses to ONE family

Per-dataset adjusted-edge-Jaccard (champion → learned-detect+champion-link), best threshold **det=0.999**:

| Family | class | champion | learned-det + champ-link | Δ |
|---|---|---|---|---|
| 44b6_0113de3b | sparse | 0.9078 | **0.9672** (tp48/fp0/fn2) | **+0.0594** ✅ |
| 44b6_0b24845f | sparse | 0.6938 | 0.6737 (tp34/fp4/fn15) | **−0.0201** ❌ |
| 6bba_05b6850b | dense | 0.5748 | **0.8661** (tp785/fp66/fn60) | **+0.2913** ✅ |
| 6bba_05db0fb1 | dense | 0.7477 | 0.8067 (tp1040/fp118/fn143) | **+0.0590** ✅ |
| **micro_adj** | | **0.6760** | **0.8305** | **+0.1545** |

Threshold sweep (micro_adj / non-regression / per-family Δ):

| thr | micro | 4/4? | 44b6_0113 | 44b6_0b24 | 6bba_05b6 | 6bba_05db |
|---|---|---|---|---|---|---|
| 0.5   | 0.8118 | no | +0.0031 | −0.0293 | +0.2362 | +0.0672 |
| 0.9   | 0.8179 | no | −0.0152 | −0.0291 | +0.2501 | +0.0690 |
| 0.99  | 0.8344 | no | −0.0056 | −0.0254 | +0.2727 | +0.0821 |
| 0.999 | 0.8305 | no | **+0.0594** | −0.0201 | +0.2913 | +0.0590 |

## Interpretation

1. **The DETECTION substrate is the genuine upside lever — not learned linking.** Holding linking classical,
   the learned detection alone lifts micro by +0.13…+0.16 and improves the dense 6bba families by +0.06…+0.29.
   Because linking is byte-identical to the champion, this gain is attributable purely to detection. This proves
   SOT-3011's post-hoc attribution ("伸び代=検出substrateで正しい") causally.
2. **Champion classical linking RECOVERS the sparse-family over-detection that sank SOT-3011.** The wholesale
   learned pipeline regressed **both** 44b6 families; here the champion's `min_track_length=4` + motion-model
   linking prunes the transient over-detections and makes **3/4** families non-regressing — including a
   near-perfect 44b6_0113de3b (adj 0.9672, fp 0) at thr=0.999.
3. **The lone obstruction is a single global detection operating point on 44b6_0b24845f.** That family
   over-detects at low threshold (det 69338 ≈ 2× champion; still 59796 after mtl=4 prune) and flips to
   under-detection at thr=0.999 (det 21119, tp34/fn15) — there is **no single global threshold** at which all
   four families clear the champion. This is the documented "single global operating point absent on sparse
   families" wall (SOT-2921/2923), now **localized to exactly one family**.
4. **Leak caveat:** the released `split_0` weights were trained on these 4 CV families, so the learned arm is a
   train-contaminated **optimistic upper bound**. The true leak-free detector is worse — this only strengthens
   the REJECT.

## Exec / kernel compatibility

**PORTABLE**, and strictly simpler than SOT-3011: detection runs offline via torch + released weights on GPU
(SOT-3011-verified attach→load→forward, `enable_internet=false`); linking is the **pure-numpy/scipy classical
champion** — the ILP/tracksdata/SCIP stack is dropped entirely. See `experiments/sot3012/exec_compat.json`.

## Disposition

- **REJECT** — 4/4 per-dataset non-regression gate fails at every global threshold; champion unchanged,
  `registry.json` unchanged, no candidate promoted, **no Kaggle submission** (child issue).
- Artifacts: `experiments/sot3012/run_ab.py`, `screen_hybrid_ab.json`, `exec_compat.json`, `ab.log`;
  ledger entry in `docs/ai/experiment_ledger.jsonl`.
- **Next axis (for the parent):** a **per-family / regime-conditional detection operating point** (peak
  threshold / `mad_k` chosen from the observable density covariate, SOT-2921/2923) applied **to the learned
  substrate**, or a **leak-free re-trained detector**, to close the single 44b6_0b24845f regression. The
  detection substrate is the confirmed lever; the remaining gap is one family's operating point, not the
  method class.
