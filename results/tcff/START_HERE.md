# START HERE

Status: **FINAL VERIFIED CANDIDATE**

Generated: 2026-07-30T03:15:35.009857+00:00

This package uses trust-signal semantics version
`2026-07-30.v2-product-intervention`. Historical raw `T` remains immutable and is
reported only as executed acceptance `A_impl`; manuscript-formula trust is
recomputed as `T_formula=G(1-U)` on the same row/support. See
`11_quality_audit/TRUST_SIGNAL_SEMANTICS_CORRECTION.md` and
`TRUST_SIGNAL_SEMANTICS_AUDIT.json` before interpreting signal results.

Confidence, entropy, Brier, NLL, and risk-coverage rows are sourced only from
`V1_P2_PROBABILITY_CORRECTED`. The historical
`P0_B_PROBABILITY_SIGNALS` bundle is retained as immutable provenance
but excluded from soft-probability claims. See
`12_v1_p2_analysis/probability_repair_audit.json`.

## Current completion

- fixed inference: 162/162
- executed-product fixed inference: 147/147
- probability runs: 39/39
- training jobs: 21/21
- trained evaluations: 315/315
- efficiency ready: True
- corrected case panels currently indexed: 26
- bootstrap replicates in this refresh: 2000
- high/critical audit failures: 0

## Read first

1. `11_quality_audit/TRUST_SIGNAL_SEMANTICS_AUDIT.json`
2. `11_quality_audit/DATA_QUALITY_REPORT.md`
3. `12_v1_p2_analysis/probability_repair_audit.json`
4. `06_summary_tables/exact_numbers_safe_to_quote.csv`
5. `06_summary_tables/Table_R1_product_fixed_corruption.csv`
6. `06_summary_tables/p0_b_signal_validity.csv`
7. `06_summary_tables/Figure_R3_source.csv`
8. `07_figures/Figure_R2_trust_validation.png`
9. `07_figures/Figure_R3_controlled_interventions.png`
10. `08_failure_cases/case_panel_index.csv`

No raw run, checkpoint, prediction, or legacy map is overwritten by this
refresh. Do not use this package as final until the status above is final and
the independent quote-ledger verifier passes.
