# CoW / TopCoW experiments — server 891

**服务器来源：西北B区 / 891机 · `63984a89ed-5af31a80` · NeurIPS论文-TOPCOW-review1**

This branch is a read-only export of the CoW and TopCoW research code from that server, with selected aggregate experiment results. It is kept separate from the 775 server branch so the two research snapshots remain identifiable. No file on the research server was changed for this publication.

## Contents

| Path | Contents |
| --- | --- |
| [`legacy-topcow/`](legacy-topcow/) | Early TopCoW training, preprocessing, baselines, independent fusion, and anatomical classification code. |
| [`tcff-base/`](tcff-base/) | Base TCFF implementation from server commit `a4ea9fb`. |
| [`tcff-final-1623e59/`](tcff-final-1623e59/) | Later review-stage TCFF implementation from server commit `8cdf91c`, including semantic postprocessing and independent audit code. |
| [`semantic-standalone/`](semantic-standalone/) | Standalone semantic analysis and verification scripts. |
| [`results/legacy-topcow/`](results/legacy-topcow/) | Configuration, training histories, and aggregate metrics for 16 completed TopCoW runs. |
| [`results/semantic-smoke-historical/`](results/semantic-smoke-historical/) | Historical July review-stage aggregate tables, R1–R3 figures, and compact quality audits. These precede the later shared final TCFF results and should not be presented as final paper claims. |
| [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) | SHA-256 hashes of the exported source files. |
| [`results/RESULTS_MANIFEST.csv`](results/RESULTS_MANIFEST.csv) | Source and published hashes of selected result files, including redaction status. |

The JSON result snapshots omit `per_case` records. Tables or figures with case identifiers are excluded. Processed medical images and labels, dataset manifests with case identifiers, model checkpoints, raw predictions, logs, and internal review documents are excluded. Large outputs and third-party pretrained weights require separate storage. These scripts retain some server-specific paths and need dataset, checkpoint, and dependency configuration before use. No project-wide reuse license is declared.

---

本分支保存 **891机** 的算法代码和筛选后的汇总结果。与默认的 775机分支分别保存，以免混淆不同服务器的实验版本。7月语义实验表格和图属于历史审稿快照，不能作为后续最终论文数值。公开结果删除了逐病例记录；原始影像、标签、模型权重和大型运行产物未上传。运行代码前需重新配置数据与权重路径。
