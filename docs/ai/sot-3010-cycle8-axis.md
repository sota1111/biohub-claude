# SOT-3010 — biohub-claude Kaggle 改善サイクル cycle-8 (explore-first) 軸選定

**Mode:** explore-first / external-knowledge-first (行き詰まり検知: 実LB順位が圏外で横ばい、champion public 0.626 vs leader 0.958).
**Decision run:** 分解のみ。Kaggle 提出なし。子Issue登録後、親は In Review で待機。

## 現状 (champion)
- champion = `detect-link-dog-v4-shorttrack-motion-gain1` (SOT-2909): **古典 DoG 検出 + ARGUS motion-model LAP linking**, 純 numpy/scipy/CPU/offline.
- leak-free CV micro_adj edge-Jaccard = **0.6760**、public LB = **0.626**、順位 = 圏外(top20未満)、LB首位 = 0.958 (差 ~0.33)。

## 前サイクル (SOT-2992, cycle-7) の確定事実（再試行禁止 / 新根拠要）
- **自前ゼロ学習 3D U-Net 検出器 (masked sparse loss, SOT-2993) = REJECTED** — micro_adj 0.6217 < 0.6760、4/4 family 退行。機構(PU汚染解消)は的中もfrom-scratch substrateが古典未満。
- **学習 cross-attention edge linking (SOT-2994) = REJECTED** — 学習linking軸は3度目の飽和 (SOT-2841/2870/2994)。伸び代はlinkingでなく検出側。
- **半教師 dense pseudo-label (SOT-2996) = REJECTED** — 機構的中もpromotable動作点ゼロ。
- **公式scorer忠実性 (SOT-2995) = 乖離0** — oracle drift は棄却済み（CV採点系は公式と桁一致）。

## 今サイクルの新根拠 (web調査・出典つき)
- 公式 royerlab baseline が **学習済み重みを公開**: 推論ノート `thibautgoldsborough/unet-baseline-inference-submission`
  (TemporalUNet3D 検出 + SimpleNodeTransformer linking)。deps = torch + tracksdata + zarr + polars + scipy、**ilpy/pyscipopt 不要**。
  組織者注: 「収束まで学習していない — より長い学習で伸びる余地」。
  → **SOT-2993 の REJECT は from-scratch/未収束が原因であって『学習検出そのもの』の否定ではない**。公開の収束済み重みを
     まるごと採用すれば、from-scratch proxy でなく実動作点で学習検出を測れる（新根拠）。
- 出典: https://github.com/royerlab/kaggle-cell-tracking-competition ,
  https://www.kaggle.com/code/thibautgoldsborough/unet-baseline-inference-submission ,
  公開ノート群 (xiaoleilian classical-baseline / ct-mix-divaug, kaiwalyaatulraut solution, pilkwang EDA/baseline, harshitsama scoring).

## 選定した独立方向ポートフォリオ (4本・構造的に独立・現champion変種にしない)
1. **role A' wholesale** — 公式学習pipelineを**公開重みごと**採用 (torch+tracksdata をoffline同梱)。full detect+link。
2. **role A' hybrid (handoff軸#3)** — 学習**検出substrateのみ** → champion古典 motion-link に接続 (linking据置き)。検出単独レバーの切り分け。
3. **role A/C portable-classical** — 公開古典/EDAノートから**可搬(numpy/scipy)な検出レバー**(正規化/閾値/前処理/augment)を蒸留移植。提出kernelリスク0。
4. **role B transfer-trust** — 公開ノート/公式baseの**検証(holdout)設計・既知leakチェックリスト**を移植し、学習候補(1-2)昇格判定用のleak-free CVを堅牢化。

## 規律 (全子に継承)
- 一次KPI = **leak-free CV** (private一次代理)。public は二次 sanity / 反証器。**public-best選抜禁止 (rogii過学習全滅の再発防止)**。
- 昇格 = 二信号一致ゲート (CV↑ を noise幅超え & public 非矛盾) — SOT-2909/kaggle-private-anchored-improvement。
- 各方向は**それ自身の実スコア/CVで評価**。非昇格時 revert + docs。昇格時 exec互換(offline kernel)を必須。
- **子IssueはKaggle提出を実行しない**。提出判定は親の再開runのみ。
- rejected軸の再試行は新根拠明示 (単一結合実測/推測での REJECT 禁止 → 証拠なければ inconclusive 止まり)。
