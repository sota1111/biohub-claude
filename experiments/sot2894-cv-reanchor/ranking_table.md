| config | issue | micro_adj (legacy) | micro_raw | lineage_macro_raw (re-anchor) | node-penalty | public LB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| detect-link-v1 | SOT-1983 | 0.3598 | 0.3419 | **0.4077** | +0.0179 | 0.509 |
| detect-link-dog-v2 | SOT-2272 | 0.5225 | 0.6561 | **0.7258** | -0.1336 | 0.5 |
| detect-link-dog-v3-adaptive | SOT-2307 | 0.6232 | 0.6531 | **0.7198** | -0.0299 | — |
| detect-link-dog-v4-shorttrack | SOT-2369 | 0.6649 | 0.684 | **0.7358** | -0.0191 | 0.624 |

Order-consistency vs same-metric public anchors (Spearman ρ; excludes unconfirmed v3):

| statistic | ρ vs public | champion is CV-top? |
| --- | ---: | :---: |
| micro_adj | 0.5 | True |
| micro_raw | 0.5 | True |
| lineage_macro_adj | 0.5 | True |
| lineage_macro_raw | 0.5 | True |
