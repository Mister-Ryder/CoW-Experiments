# CoW experiments / 脑动脉环实验

Research code for Circle of Willis (CoW) segmentation and the TCFF foundation-feature experiments. This repository contains a source snapshot and selected aggregate results from the research server on **2026-10-13**.

## Contents

| Directory | Contents |
| --- | --- |
| [`tcff/`](tcff/) | Current TCFF training, evaluation, preprocessing, and rebuttal analysis code. Start with [`REBUTTAL_PROJECT_README.md`](tcff/REBUTTAL_PROJECT_README.md). |
| [`legacy-topcow/`](legacy-topcow/) | Earlier TopCoW experiment scripts retained for provenance. |
| [`results/tcff/`](results/tcff/) | Selected aggregate metrics, figures, configuration, and training history; case-level records are excluded. |
| [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) | SHA-256 hashes for the copied source snapshot. |

The `tcff/` snapshot includes the server working tree's modified code and seven new v2 analysis/control scripts. It is an exported snapshot rather than a reconstruction of the server's Git history. The earlier scripts are kept separately so they do not replace the current implementation.

## Running the code

The main entry points are `tcff/topcow_train_foundation.py` for single-GPU training, `tcff/topcow_train_foundation_ddp.py` for multi-GPU training, `tcff/topcow_eval_official_all.py` for evaluation, and `tcff/tcff_fdd_final_v7_single_panel_neurips.py` for the FDD analysis. The model and baseline implementations are in `tcff/topcow_foundation_model.py`.

The server environment was not pinned to a dependency lockfile. Before running, install compatible PyTorch and imaging dependencies, obtain the upstream SAM-Med3D code separately, and configure dataset/checkpoint locations for your environment. Several original scripts still contain absolute paths from the research server. The repository does not include the SAM-Med3D source or pretrained checkpoint.

## Data and result scope

Processed medical images, labels, model weights, raw predictions, training logs, and case-level metrics are excluded from this public repository. Some outputs exceed GitHub's file limits, and redistribution rights for the datasets and pretrained weights have not been established here. The selected result files are intended to document aggregate findings; they are not a replacement for the full research archive. No project-wide reuse license is declared.

---

本仓库保存 CoW/TCFF 算法代码、早期 TopCoW 脚本和筛选后的汇总实验结果。服务器快照包含当时尚未提交的代码修改；源码哈希见 `SOURCE_MANIFEST.json`。原始影像、标签、模型权重、逐病例结果与大型运行产物不在公开仓库中，需通过独立备份保存。运行前需要自行配置数据和权重路径，以及 SAM-Med3D 等依赖。
