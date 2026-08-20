# SOT-2847 — Submission-constraint verdict: GPU + attached offline weights ARE feasible

**Cycle 8, child A (parent SOT-2846). Escalation-ladder step 3 (generalization-gap
diagnostic). No Kaggle submission. Champion `champion/config.json` byte-frozen.**

## TL;DR verdict

**PORTABLE / FEASIBLE.** The champion config's long-standing premise — that a
GPU-pretrained-weights model *"cannot run under this repo's numpy/scipy/zarr, CPU,
no-internet, no-weights kernel"* — is **false as an infrastructure constraint**.
`biohub-cell-tracking-during-development` is a Kaggle **Code competition** whose
submission kernels **can enable GPU** and **read pretrained weights from an
attached Kaggle Dataset / Model offline** (internet off). Public learned-inference
notebooks already submit trained models in exactly this competition. The
classical→learned score gap (~0.62 → ~0.89) is therefore a **detector-quality gap,
not a platform limitation** — the pivot to a learned detector (children SOT-2848 /
SOT-2849) is unblocked.

The "no-weights/no-GPU/CPU-only" wording remains an accurate description of the
*current champion kernel's own metadata* (`enable_gpu: false`, no `model_sources`),
but it is **not a competition constraint**. It was a self-imposed default, not a
rule.

## What was verified, and how

### 1. Kaggle Code-competition mechanics (rules layer)

Kaggle Code competitions accept a submission only from a **notebook/script kernel**
run against the hidden test set (a raw `submission.csv` upload is rejected — the
constraint SOT-1984 already encodes). For that kernel:

- **Internet must be OFF** to be submission-eligible (`enable_internet: false`).
- **GPU is available** (T4×2 / P100) and selectable per kernel (`enable_gpu: true`).
- **Pretrained weights are supplied offline** by attaching them as a **Kaggle
  Dataset** or **Model** source; the kernel `torch.load`s them from
  `/kaggle/input/<slug>/…` with no network. This "train in an internet-on kernel →
  save weights as a Dataset → attach to an internet-off inference kernel" chain is
  the standard offline-submission pattern.

Sources:
- Kaggle competition page — Biohub - Cell Tracking During Development:
  https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
- Offline weights-attach pattern (train-kernel → dataset → offline inference kernel):
  https://towardsdatascience.com/easy-kaggle-offline-submission-with-chaining-kernels-30bba5ea5c4d/
- Kaggle Code-competition submission requirements (internet off; run top-to-bottom;
  ≤9h GPU/CPU): https://www.kaggle.com/discussions/questions-and-answers/498601

### 2. This competition's own learned notebooks (existence proof)

Trained models demonstrably submit **in this competition**, on GPU, reading
attached offline weights:

- **royerlab official baseline** — `TemporalUNet3D` (3D U-Net + temporal attention)
  detection producing a per-voxel single-channel detection map (cell centres via
  local-max suppression) + `SimpleNodeTransformer` cross-attention linking, with a
  released weights checkpoint used by the public UNet-baseline **inference** notebook:
  https://github.com/royerlab/kaggle-cell-tracking-competition
- **Public UNet baseline inference submission** (thibautgoldsborough) — proves a
  trained model submits in this competition (GPU, attached weights, internet off).
- **pilkwang — learned-graph w/ gap recovery**, public ≈0.890:
  https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery
- Additional learned/scoring notebooks: anhadmahajan06, kaiwalyaatulraut, harshitsama;
  classical baseline (xiaoleilian) caps ~0.62 — the ceiling this repo already sits at.

Example submissions in this competition run on **GPU T4×2**, confirming GPU
availability for the scored kernel.

### 3. In-container capability (execution layer)

- `torch 2.5.1+cu121`, CUDA available on an **RTX 3080 Ti**; `data/train` (GT `.geff`)
  and `data/test` (1.8 GB `.zarr`) present. GPU training/inference is available
  locally for the detector port (child SOT-2848).

### 4. Offline attach → load → forward — **measured**, not assumed

The acceptance-critical claim is the *offline weights path itself*. A self-contained
smoke test (`biohub_tracking.learned_detect.run_weights_attach_smoke`) exercises it
end to end: build a tiny torch detection head → **save** its `state_dict` under a
simulated Kaggle Dataset mount `…/kaggle_input/biohub-claude-weights/detector.pt` →
resolve it with the same discovery the kernel uses → **`torch.load` it offline** →
**forward** over a synthetic `(T,Z,Y,X)` volume → recover centroids.

Measured result (`experiments/sot2847/weights_attach_smoke.json`, GPU run):

```json
{
  "torch_version": "2.5.1+cu121", "cuda_available": true, "device_used": "cuda",
  "weights_path": ".../kaggle_input/biohub-claude-weights/detector.pt",
  "resolved_offline": true, "state_dict_bit_exact": true, "forward_ok": true,
  "num_timepoints": 3, "total_detections": 693,
  "verdict": "offline attach->load->forward FEASIBLE"
}
```

The reloaded weights are **bit-exact** to the saved ones and the forward pass
produces detections — the offline attach path a learned submission kernel needs
**works**.

## What this issue delivers (the receptacle)

Default-off, so the champion is byte-for-byte unchanged (config sha256
`42064648e612183e761bf9d40b70d3e8a2497453a878f1a44f5b52e410e01bdd`, CV micro-adj
`0.6649` reproduced; exec-compat gate green):

- **`src/biohub_tracking/learned_detect.py`** — a torch `LearnedDetector` matching
  the classical detector contract (`detect_centroids(volume) → (N,3)`,
  `detect_series(arr, max_t) → {t:(N,3)}`), Kaggle-mount weights discovery, offline
  `state_dict` load, and the weights-attach smoke. torch is imported **lazily** so
  the classical champion / CI / exec-gate keep running torch-free.
- **Pipeline switch** — `run_pipeline(..., learned_detector=None)`; a built detector
  replaces the detect stage, the linker/scorer/submission stages are untouched.
  `None` (the only value the champion supplies) is byte-identical to before.
- **Config passthrough** — `champion.learned_detector_config(config)` reads an
  optional `learned_detector` block (top-level or under `detect`); the champion has
  none → classical path. Honoured by `build_submission` and the CV harness.
- **Learned kernel build path** — `build_kernel.build(..., learned=True,
  model_sources=[...])` embeds the torch module, sets `enable_gpu: true` and the
  attached `model_sources` while keeping `enable_internet: false`. Champion/candidate
  builds are unchanged (CPU, no torch, no model sources).

This is **infrastructure, not a promotion**: nothing here changes the submitted
champion. The trained-detector port and its leak-free CV A/B live in **SOT-2848**
(detector) and **SOT-2849** (learned linking / gap recovery).

## Caveats / residual risk (honest)

- The receptacle ships a **placeholder** arch (`voxel_scorer_v0`) only to prove the
  plumbing; it is **not** the frontier detector and makes no score claim. Real weights
  + normalization + the `TemporalUNet3D` architecture are SOT-2848's job.
- The verdict answers *"can a learned model run in the submission kernel?"* (yes). It
  does **not** claim the learned pipeline will beat the champion on the leak-free CV —
  that is measured downstream under the same-seed no-per-family-regression gate.
- The ≤9h kernel runtime budget for a full 3D-U-Net inference over the hidden test set
  is a downstream engineering constraint for SOT-2848, not a blocker to this receptacle.
