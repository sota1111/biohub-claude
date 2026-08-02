# SOT-2307 — Dataset-robust adaptive-intensity detection

Kaggle 順位向上サイクル第4次（親 SOT-2300）の子。escalation ladder の「アーキテクチャ変更」段。
DoG-v2 champion がデータセット間の強度スケール差に頑健でない根本原因に対処する。

## 真因

champion `detect-link-dog-v2`（SOT-2272）は Difference-of-Gaussians 応答の **固定 percentile(92)**
を閾値にする。percentile 閾値は「一定の voxel **割合**」を残すため、実細胞の密度／体積比がデータセット
ごとに違うと検出「数」がデータセット依存になる。SOT-2305 の4データセット holdout で per-dataset を見ると：

| dataset | n_true | DoG-v2 pred_nodes | DoG-v2 adj |
| --- | ---: | ---: | ---: |
| 44b6_0113de3b | 25755 | 30691 | 0.8865 |
| 44b6_0b24845f | 32795 | 52213 | 0.662 |
| **6bba_05b6850b** | **6362** | **40450** | **0.2622** |
| 6bba_05db0fb1 | 69800 | 74282 | 0.7141 |

疎な `6bba_05b6850b` は真細胞 6362 に対し 40450 ノードを予測＝約6倍の過検出。ノード数ペナルティ
`adj = J·max(0, 1 − 0.1·(N_pred−N_true)/N_true)` により adj が 0.262 に潰れ、holdout micro-adj
0.5225 の主たる押し下げ要因になっていた。

## 対策軸（適応的強度正規化）

DoG 応答に対する **ロバスト per-volume 適応閾値** を `DetectParams.mad_k` として追加
（`src/biohub_tracking/detect.py`）:

```
threshold = median(response) + mad_k · 1.4826 · MAD(response)
```

`1.4826·MAD`（中央絶対偏差）は応答標準偏差の外れ値ロバスト推定。閾値を各ボリューム自身の
ノイズフロアに合わせるので、固定割合ではなく**信号量に検出数を適応**させる。`mad_k=None` は
従来の percentile 動作を完全に保持（後方互換）。

## screen → confirm

- **screen**（`experiments/sot2307/screen_adaptive.py`）: DoG 応答＋局所極大を per-family/timepoint に
  一度だけ計算してキャッシュし、`mad_k` を安価に sweep。キャッシュ経路は実 `detect_centroids(mad_k=k)`
  と1タイムポイントで完全一致を確認済み。

  | mad_k | holdout micro-adj | 44b6 | 6bba |
  | ---: | ---: | ---: | ---: |
  | 2 | 0.6016 | 0.7415 | 0.5954 |
  | **3** | **0.6232** | **0.7716** | **0.6168** |
  | 4 | 0.5979 | 0.7655 | 0.5907 |
  | 5 | 0.5527 | 0.7656 | 0.5435 |
  | 6 | 0.4866 | 0.7225 | 0.4763 |
  | 8 | 0.3732 | 0.5028 | 0.3676 |
  | 10 | 0.3151 | 0.4599 | 0.3089 |

  `k∈{2,3,4}` がいずれも champion(0.5225)超＝上振れ1点ではなく plateau。peak は k=3。

- **confirm**（`experiments/sot2307/confirm_adaptive.py`）: 勝者 `mad_k=3.0` を**実 `run_pipeline`**
  （キャッシュ近道でなく本番検出経路）で 4-family 独立再採点。champion も同一ハーネスで並置採点。

  | detector | holdout micro-adj | 44b6 | 6bba |
  | --- | ---: | ---: | ---: |
  | incumbent DoG-v2 (pct92) | 0.5225 | 0.7721 | 0.5117 |
  | **adaptive mad_k=3** | **0.6232** | 0.7716 | 0.6168 |
  | Δ | **+0.1007** | −0.0005 | +0.1051 |

  per-dataset（confirm, mad_k=3）:

  | dataset | adj | prec | rec | pred_nodes |
  | --- | ---: | ---: | ---: | ---: |
  | 44b6_0113de3b | 0.8814 | 0.9592 | 0.94 | 32147 |
  | 44b6_0b24845f | 0.6658 | 0.881 | 0.7551 | 42080 |
  | 6bba_05b6850b | 0.5025 | 0.7115 | 0.7325 | 13376 |
  | 6bba_05db0fb1 | 0.7096 | 0.853 | 0.814 | 73953 |

`6bba_05b6850b` の matched-edge TP/FP/FN は **619/251/226 のまま不変**（precision/recall も不変）。
利得は全て約27000の孤立過検出を刈り込んだことによるノード数ペナルティの解消＝マッチング指標の
アーティファクトではない honest な改善。per-dataset の precision/recall はデータセット間で乖離せず
（44b6 prec 0.88–0.96 / rec 0.75–0.94、6bba prec 0.71–0.85 / rec 0.73–0.81）、どのデータセットも
崩壊しない。

## 判定

**昇格（promoted）**。champion を `detect-link-dog-v3-adaptive` に更新（`champion/config.json` に
`mad_k: 3.0` 追加、`registry.json` 昇格、`champion.py` EMBEDDED_CHAMPION_CONFIG 同期）。
exec互換ゲート＋pytest 55件 pass。Kaggle 提出は行っていない（screen 専用）。

## 留意（オラクル残差）

SOT-2305 の記録どおり、4-family holdout はまだ完全な LB proxy ではない（v1 の holdout 0.36 が
自身の LB 0.509 を再現しない）。本改善は「疎データセット過検出」という LB 劣化の特定原因を直接
是正し全データセットで非退行なので順位改善が期待できるが、実 LB での確認は次サイクルの提出判断
（converge/締切に応じて）に委ねる。今回は提出禁止指示に従い screen 昇格のみ。
