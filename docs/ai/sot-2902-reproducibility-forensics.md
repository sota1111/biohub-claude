# SOT-2902 — Champion reproducibility forensics: public 0.624 (48b1e) vs 0.509 (e445965)

**Type:** diagnostic / reproducibility forensics (no champion change, no submission, no promotion).
**Question (parent SOT-2901 premise):** `champion/config.json` (`detect-link-dog-v4-shorttrack`,
SOT-2369) is described as producing public **0.624** (artifact `48b1e…`) but now "regenerates" a
stable public-**0.509** artifact (`e445965…`). Is the 0.624→0.509 drop **(a) code regression**,
**(b) config drift**, or **(c) judge-stochasticity / metric-patch**?

**Reproduce:** `PYTHONPATH=src python experiments/sot2902/confirm_restore_0624.py` (byte + CV +
exec-compat, writes `experiments/sot2902/confirm_restore_0624.json`);
`PYTHONPATH=src python -m biohub_tracking.eval.cv --check-champion` (champion CV guard →
`experiments/sot2902/champion_cv_repro.json`).

---

## Bottom line

**The premise contains a fingerprint misreading. `48b1e` and `e445965` are NOT two submission
artifacts — they are two different fingerprint *schemes* emitted by the control-plane submit script
`scripts/ai/kaggle_targets_submit.sh` for the same byte-frozen champion.** HEAD reproduces `48b1e`
(the 0.624 source) exactly; it never regenerates `e445965`, which is not a content hash at all.

| Verdict | Classification |
| --- | --- |
| (a) code regression (target repo) | **RULED OUT** — kernel `.py` byte-frozen since the 0.624 commit; HEAD sha == `48b1e`. |
| (b) config drift (target repo) | **RULED OUT** — `champion/config.json` byte-frozen (`42064648…`). |
| (c) real 0.624→0.509 Kaggle score gap | **INCONCLUSIVE** — metric-patch re-score vs control-plane submit-version drift; not decidable without Kaggle notebook version-history + a (forbidden) re-submission. |

## 1. What `48b1e` and `e445965` actually hash

Both come from the fingerprint block in `scripts/ai/kaggle_targets_submit.sh`, which picks one of
several branches depending on the registry `submit` spec and whether a local `submission.csv` exists:

| Submission (ref, date, score) | Recorded `artifact:sha256` | What it actually hashes |
| --- | --- | --- |
| 55212214 · 2026-08-03 · **0.624** | `48b1eaa2dfb63c8a…` | `sha256(submit/kernel/biohub-claude-champion.py)` — a **real source-file hash** |
| 55649179 · 2026-08-20 · **0.509** | `e445965c6cdea076…` | `sha256("kaggle-notebook:sota1111/biohub-claude-champion@4:submission.csv")` — a **synthetic fallback identity** |
| 55193790 · 2026-08-02 · 0.557 | `01c2f3938f8ae7fa…` | `sha256("kaggle-notebook:…@7:submission.csv")` — fallback identity, version **7** |
| — · 2026-08-02 · 0.500 | `a2c9a7681f2c98…` | `sha256("kaggle-notebook:…@5:submission.csv")` — fallback identity, version **5** |

The fallback identity (`kaggle-notebook:<kernel>@<version>:<output>`) encodes **only** the kernel
name, a hand-maintained registry `version`, and the output filename — **never the scored
`submission.csv` content**. So `e445965` cannot be "regenerated" or "matched" by any local build; it
is a label, not an artifact hash. All three fallback fingerprints (`e445965`/`01c2f393`/`a2c9a7`)
were verified by direct reconstruction (`experiments/sot2902/confirm_restore_0624.py`,
`e445965_is_kaggle_notebook_fallback_identity = true`).

**Consequence:** the "`48b1e` → `e445965` artifact change" is a **fingerprint-scheme flip** (real
file-sha branch → version-only fallback branch, driven by registry `submit.file`/`version` edits),
not a change in the submitted computation. The two values are **not comparable** as artifact
identities.

## 2. HEAD reproduces the 0.624 source, byte-for-byte

`submit/kernel/biohub-claude-champion.py` — the submission-generating source — has been **byte-frozen
since the SOT-2369 commit `57bdf4f` (2026-08-03 12:50) that produced the 0.624 submission**:

```
57bdf4f  2026-08-03  sha256 48b1eaa2…   (dog-v4-shorttrack — the 0.624 source)   ← HEAD == this
fd9f4a3  2026-08-02  sha256 8e8354fc…   (dog-v3-adaptive)
c93daa7  2026-08-02  sha256 1c4e14a2…   (dog-v2)
166fdd8  2026-07-26  sha256 c169ac0e…   (v1 baseline)
```

`sha256sum submit/kernel/biohub-claude-champion.py` at HEAD = `48b1eaa2…` = the 0.624 artifact
fingerprint. `champion/config.json` is byte-frozen at `42064648…` over the same window.
**→ HEAD regenerates `48b1e`, not `e445965` (AC #2).**

## 3. Restore candidate for the 0.624 config

Because the 0.624 config was never lost, the restore candidate is a frozen snapshot of the
byte-identical champion params: `champion/candidates/sot2902-restore-0624.json`.

- **Effective config identical to champion:** `true` (every detect + link knob matches).
- **exec-compat:** the champion kernel that carries this config passes the exec-runtime gate
  (no `__file__`, undefined cwd) — `exec_compat_ok = true`.
- **Leak-free CV (SOT-2817 re-anchored full metric):** micro-adj **0.6649**, full score **0.6649**,
  division_term 0.0000 (Δ 2e-06 vs `CHAMPION_REFERENCE`). **→ AC #3.**

Per-dataset (unchanged from the frozen champion): 44b6_0113de3b 0.8895 · 44b6_0b24845f 0.6817 ·
6bba_05b6850b 0.5700 · 6bba_05db0fb1 0.7310. **No Kaggle submission** (parent-resume's decision).

## 4. The residual real score gap (inconclusive, per the issue's own bar)

The one thing byte-forensics cannot settle is the actual 0.624→0.509 number on Kaggle's hidden set,
because neither scored `submission.csv` is stored (the fingerprints prove the CSV content was never
even hashed). Two non-exclusive mechanisms remain, neither decidable from inside the container:

1. **Metric-patch re-score of a frozen artifact** (SOT-2816's conclusion): the whole leaderboard was
   re-scored in the 08-02→08-20 window — `docs/ai/kaggle/leaderboard-rank.jsonl` topScore
   **0.943 → 0.957** — consistent with the documented division-jaccard exploit patch. A
   division-forfeiting (`allow_division=false`) champion's number moves under such a re-score with no
   code change.
2. **Control-plane submit-version drift**: the registry `submit.version` used for the fallback
   identity fell from **7** (2026-08-02) to **4** (2026-08-20), so the auto-submit's
   `kaggle competitions submit -v 4` may have pushed an **older** Kaggle notebook version than the
   0.624 run. This is an operational drift in the control-plane registry, not the target repo.

Deciding between them requires Kaggle notebook version-history plus a re-submission of the identical
artifact — the latter forbidden by this issue — so the **magnitude split is inconclusive** (the
issue instructs: don't assert from a single coupled observation → record inconclusive).

## 5. Follow-up (control-plane, out of this target-repo's scope)

`scripts/ai/kaggle_targets_submit.sh` should emit **one stable fingerprint scheme** and never the
version-only `kaggle-notebook:@<version>` fallback identity for a code-competition target, so a
byte-frozen champion can never present as a "new artifact" (and, conversely, a genuinely new build is
never false-deduped). Tracked as a recommendation on SOT-2901 / control-plane; not implemented here
because SOT-2902's change scope is the `biohub-claude` target repo.
