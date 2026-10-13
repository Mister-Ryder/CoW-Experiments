# Reviewer evidence map

Generated: 2026-07-30T03:15:26.938558+00:00

This file maps one unified evidence system to each reviewer; it does not create reviewer-specific experimental conclusions.

## Area Chair

- Reliability and robustness: Figures R1-R3; `06_summary_tables/p0_a_utility_summary.csv`, `p0_b_signal_validity.csv`, and `p0_c_degradation_slopes.csv`.
- Component contribution and simple alternatives: `06_summary_tables/Table_R1_components.csv`, `Table_R1_rebuttal.csv`, and `Table_R1_product_fixed_corruption.csv`.
- Reliability of reported numbers: `11_quality_audit/DATA_QUALITY_REPORT.md`, `ANALYSIS_VALIDATION_REPORT.md`, and the independently verified quote ledger.
- Efficiency: `09_efficiency/efficiency_summary.csv` and `06_summary_tables/Table_R2_efficiency.csv` when present.

## Rkxt

- Whether TCFF is more than a renamed gate: paired update utility (Figure R1), explicit `1-U`/`G`/`T_formula`/`A_impl` construct tests (Figure R2), fixed and retrained alternatives (Table R1), and frozen-threshold HAR/BRR.
- Falsifiability: Figure R4 retains incorrect acceptance and over-suppression; no missing common-mode run is inferred.
- Formula scope: `T_formula=G(1-U)` is tested as a diagnostic score on historical runs; performance of the product rule comes only from separately labelled runs in which `A_product=G(1-U)` is actually executed.

## pCat

- Proxy stability and shift: source-versus-external rows in Figure R1/R2 source tables and controlled corruption curves in Figure R3.
- Probability controls: confidence and entropy rows in `p0_b_signal_validity.csv` when the dedicated probability matrix completes.
- Cost: the same-device, batch-1, FP32 efficiency manifest and Table R2.

## mDyq

- Mild-to-severe auxiliary corruption: Gaussian, channel dropout, spatial translation, and donor mixing are kept in separate physical/statistical units.
- Ablation and engineering evidence: Table R1, run manifests, three-seed retraining coverage, and Table R2.

## Completion snapshot

- Fixed-checkpoint runs: 162/162 successful.
- Executed-product fixed-checkpoint runs: 147/147 successful.
- Training jobs: 21/21 successful.
- Trained evaluation runs: 315/315 successful.
- Bootstrap replicates in this refresh: 2000.
