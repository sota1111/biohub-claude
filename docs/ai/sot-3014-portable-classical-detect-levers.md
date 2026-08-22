# SOT-3014 — Portable classical detection levers distilled from public notebooks

**Cycle 3 (SOT-3010), direction 3 · role A/C portable-classical · explore-first.**
Verdict: **REJECT** (real count-neutral CV gain, but fails the strict 4/4
per-dataset non-regression gate — family-mix-sensitive). No Kaggle submission.

## Goal

Read this competition's own **public classical / EDA notebooks**, distill any
*portable* (numpy/scipy/CPU/offline, no learned weights) detection lever that is a
**new mechanism** vs the axes already rejected across cycles 2–5, apply it to the
champion DoG detector, and A/B it on the same-seed leak-free CV. Promotion only on
the two-signal gate (CV non-regression + non-contradicting public LB); this child
does **not** submit.

## Notebooks read (Kaggle-authenticated pull → `experiments/sot3014/notebooks/`)

| Notebook | Portable content | Verdict for us |
| --- | --- | --- |
| `xiaoleilian/biohub-cell-tracking-classical-baseline` | Pure numpy/scipy/skimage classical pipeline: XY block-pool downsample, Otsu∨relative threshold, **intensity-weighted sub-voxel centroid refine (`_refine`)**, physical-µm cKDTree NMS, two-pass motion Hungarian link. | Source of the tested lever (see below). Its other knobs (XY downsample, Otsu/relative threshold) are *count/threshold* mechanisms already in the rejected operating-point family. |
| `kaiwalyaatulraut/biohub-cell-tracking-solution` | Header sets classical knobs (`THRESH_REL`, `peak_local_max`, Otsu) but the actual pipeline is the **learned TemporalUNet3D detect-head + ILP linker** on offline GPU wheels. | Non-portable (GPU weights) — this is the learned-detector frontier already covered by SOT-2993/3011, not a classical lever. |
| `pilkwang/…-data-model-eda-baseline` | EDA + the support-pack (offline wheels + 50ep weights) the learned notebooks import. | No new portable detection mechanism. |
| `harshitsama/biohub-scoring-data-fully-explained` | Exact scoring metric: edge Jaccard × node-count adjustment `(1 − 0.1·(N_pred−N_est)/N_est)` + 0.1·division Jaccard; GT is **sparse**. | Confirms our re-anchored CV metric; no detection lever. |
| `xiaoleilian/biohub-ct-mix-divaug` | mix / division-augmentation for a **learned** model (training-time). | Not train-free / portable at inference. |

**Distilled portable lever (new mechanism):** every notebook's train-free
detection idea is either (a) already in this repo's rejected operating-point ladder
(threshold / count / normalization) or (b) a learned-weight pipeline. The one
mechanism that is **portable AND not a rejected axis** is `xiaoleilian`'s
`_refine`: **intensity-weighted sub-voxel centroid refinement**. Every prior
detection lever changes *which / how many* detections; none refine *where* an
accepted detection sits. Implemented as default-off `DetectParams.subvoxel_refine =
(rz, ry, rx)` (`src/biohub_tracking/detect.py`): after all count-changing stages,
each kept centroid is replaced by the centre-of-mass of the background-subtracted
normalized volume in the local window — **count-neutral by construction**, so
recall and the node-count penalty are untouched; only the ≤7 µm node matching and
the motion-model linking velocities can change.

## Same-seed leak-free CV A/B (SOT-2761 4-family holdout, re-anchored micro_adj)

Frozen downstream = champion `detect-link-dog-v4-shorttrack-motion-gain1`
(motion-link gain1 + short-track mtl=4). Incumbent micro-adj = **0.6760**.
Full machine table: `experiments/sot3014/screen_subvoxel_refine.json`.

| variant `(rz,ry,rx)` | 44b6_0113de3b | 44b6_0b24845f | 6bba_05b6850b | 6bba_05db0fb1 | micro-adj | Δ | 4/4 non-reg | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **baseline (champion)** | 0.9078 | 0.6938 | 0.5748 | 0.7477 | 0.6760 | — | — | — |
| refine (1,2,2) | 0.9078 | 0.6878 ▼ | 0.5821 | 0.7505 | 0.6802 | +0.0042 | ✗ | REJECT |
| refine (1,3,3) | 0.9077 ▼ | 0.6748 ▼ | 0.5787 | 0.7473 ▼ | 0.6767 | +0.0007 | ✗ | REJECT |
| refine (2,5,5) | 0.9077 ▼ | 0.6876 ▼ | 0.5971 | 0.7552 | 0.6891 | +0.0131 | ✗ | REJECT |
| refine (2,3,3) | 0.9077 ▼ | 0.6876 ▼ | 0.6022 | 0.7518 | 0.6898 | +0.0138 | ✗ | REJECT |

## Reading

- The lever is **real**: every window raises the aggregate micro-adj, and the best
  (`2,3,3`, +0.0138) is well past CV noise — sub-voxel centroids genuinely tighten
  the dense-family matching and linking.
- But the gain is **entirely on the dense `6bba` families** (05b6850b
  0.5748→0.6022, 05db0fb1 0.7477→0.7552), while **both sparse `44b6` families
  regress slightly** (0113de3b 0.9078→0.9077, 0b24845f 0.6938→0.6876). Refining the
  *single* sparse tracked centroid over a multi-voxel window occasionally nudges it
  enough to break one short-track edge. This is the same **family-mix wall**
  documented across biohub cycles: a global detection change that helps dense
  fused-nucleus families structurally cannot leave the isolated sparse nuclei
  untouched.
- Under the mandatory **per-dataset non-regression** discipline (only four videos —
  selecting on the micro alone overfits the holdout), every variant is **REJECT**.
  Count-neutrality is confirmed (detection count identical; only the linked/pruned
  node count shifts through linking).

## Disposition

- **No promotion, no submission.** `champion/config.json` is unchanged
  (byte-identical); the lever ships only as default-off `DetectParams.subvoxel_refine`
  infra, matching the repo convention for rejected levers (blobness_filter,
  local_threshold, recall_recovery, …).
- Escalation-ladder note: the *portable-classical operating-point* axis is now
  exhausted from a new angle (position refinement, not just count/threshold) — the
  remaining head-room stays on the **learned-detector substrate** (SOT-2993 /
  SOT-3011 lineage), consistent with the cycle-7/8 finding that biohub is
  fundamentally a learned-detector game capped for classical at CV≈0.676.
- Reproduce: `.venv/bin/python experiments/sot3014/screen_subvoxel_refine.py`
  (deterministic; same seed reproduces every score).
