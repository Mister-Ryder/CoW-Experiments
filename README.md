# CoW / TopCoW early experiment snapshot

**Server marker:** `server/autodl-f4354c9c5a-topcow-early` · verified host `autodl-container-f4354c9c5a-ccf1861e`.

This branch records a small, read-only source and aggregate-result snapshot from an earlier CoW/TopCoW research environment. The machine's area and number were not independently verified; `888-topcow-TGF` was a project directory name. The 775 and 891 server snapshots remain on their own branches.

## Contents

| Path | Contents |
| --- | --- |
| [`legacy-topcow/`](legacy-topcow/) | Early TopCoW training, preprocessing, model, and baseline scripts. |
| [`tcff-base/`](tcff-base/) | Base TCFF source snapshot at server commit `a4ea9fb`. |
| [`tcff-final-49d9c2e/`](tcff-final-49d9c2e/) | Intermediate review-stage TCFF source snapshot at commit `49d9c2e`. |
| [`results/early-foundation/`](results/early-foundation/) | Two completed historical training runs (`sam_tgf_v3_ddp`, `sam_tgf_v3_s123`) with configuration, aggregate results, and training history; three `sam_tgf_v4_nogate*` configurations have **no completed result**. |
| [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) | Published source paths, file sizes, and SHA-256 checksums. |
| [`results/RESULTS_MANIFEST.csv`](results/RESULTS_MANIFEST.csv) | Original and published result checksums, transformations, and run status. |

Compared bytewise with the public 775 and 891 branches, 95 of the 98 source files already appear there. The three distinctive intermediate-version files are `tcff-final-49d9c2e/tcff_g1_followup.py`, `tcff-final-49d9c2e/tcff_rebuttal_runner.py`, and `tcff-final-49d9c2e/topcow_foundation_model.py`. The full small-source snapshot is retained so these files can be read with their matching imports and scripts.

The two old performance-summary notes are named `HISTORICAL_实验性能汇总.md` in this branch. Their original bytes are preserved, but their "best" labels refer to an earlier stage and must not be read as final rebuttal claims.

## Trust-gate version

In the `49d9c2e` model, let `w = W_u(fused)` and `U = uncertainty`. The `full` variant executes `fused * (1 - w*U) + seg_feat * (w*U)`; its coefficient on `fused` is `1 - w*U`. **This version has no `product` variant.** The later 891 snapshot adds `product`, whose coefficient is `(1 - w) * (1 - U)`. These are distinct executed interventions; the intermediate `full` code must not be cited as an implementation of a `G(1-U)` product gate.

## Data and reuse scope

The published result JSONs omit case-level records and redact absolute server paths. The two completed runs are historical snapshots, not final paper results. Configurations without a result file are marked accordingly in the manifest.

Medical images and labels, dataset manifests, model and pretrained weights, raw predictions, logs, environment files, server instruction files, and third-party SAM-Med3D source are excluded. Reproducing the experiments requires those separately obtained inputs and a compatible software environment. Several original scripts retain server-specific path assumptions. No project-wide reuse license is declared.

---

本分支标记为 `server/autodl-f4354c9c5a-topcow-early`，保存前期算法和少量历史汇总结果。两项训练已有结果，另外三项仅有配置，不能当作完成实验。中间版 `full` 的实际融合系数为 `1-wU`，该版本没有 `product` 消融。逐病例记录、影像、权重与日志均未公开。
