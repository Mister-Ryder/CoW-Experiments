# TCFF CoW Rebuttal Project

本目录由 `topcow-TGF` 只读整理而来，仅保留论文中 Circle of Willis（CoW）/foundation-feature 实验链路及后续 rebuttal 所需内容。

## 论文结果对应关系

- Lausanne 的 TCFF 结果：`Runs_foundation/sam_tgf_v3`，seed 42。
- IXI-HH 的 TCFF 结果：`Runs_foundation_v2/sam_tgf_v3_s0`，seed 0。
- 对比方法：Concat、BAGF、DSFF、Cross-Attention、AFF。
- 论文 Figure 1：`analysis_fdd_foundation_all_ixi_vis3`。
- 论文 Table 3/12 数值汇总：`实验性能汇总.md` 与各运行目录下的 `official_metrics*.json`。

## 数据目录用途

- `TopCoW_processed`：125 例训练集。
- `crown_processed`：原训练脚本用于评估和选择 `best_model.pth` 的 20 例数据。
- `lausanne_processed`：论文 Lausanne 外部测试集，20 例。
- `ixi_processed`：论文 IXI-HH 外部测试集，20 例。

## 代码入口

- `topcow_train_foundation.py`：TCFF v3 单卡训练脚本。
- `topcow_train_foundation_ddp.py`：Cross-Attention/AFF 等多卡训练脚本。
- `topcow_eval_official_all.py`：按数据集标签保存正式指标的评估脚本。
- `tcff_fdd_final_v7_single_panel_neurips.py`：论文 Figure 1 的最终 FDD 分析脚本。
- `topcow_foundation_model.py`：包含最终 TCFF、Concat、BAGF、DSFF、Cross-Attention、AFF 实现。

运行前需要将本目录下的 `SAM-Med3D` 加入 Python 模块搜索路径；原始训练脚本中仍保留服务器路径 `/root/888-topcow-TGF/SAM-Med3D`，应在新的运行环境中显式配置。

## 已排除内容

- `Runs_compare*`、`Runs_indcls*` 等非论文最终 foundation 分支。
- `sam_tgf_v4*`、CTC 过程实验和未完成的旧 Cross-Attention 运行。
- 旧版 FDD 目录，仅保留最终 `analysis_fdd_foundation_all_ixi_vis3`。
- 各运行目录的 `last_model.pth`，仅保留论文评估使用的 `best_model.pth`。
- 大规模过程可视化、预测缓存和旧日志。

## 复现注意事项

- 论文 TCFF 的 Lausanne 与 IXI-HH 数值来自不同随机种子 checkpoint。
- Cross-Attention 和 AFF 的 IXI-HH `official_metrics.json` 未在原项目中保留，需要用现有 checkpoint 重新评估。
- 正式 Table 3 评估默认使用 Z/Y 两轴翻转组成的 4-view TTA；最终 FDD 分析为非 TTA 单次推理。
- 当前项目没有锁定的 Python 环境文件或完整 Git 历史，首次运行前应单独记录环境版本。
