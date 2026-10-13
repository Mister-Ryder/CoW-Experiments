# Final Result Summary

Status: **complete**  
Generated: 2026-07-30T03:15:26.634389+00:00

## Reproducibility identity

- execution git commit(s): 0ed9605f745fab302e9761ac532c6a15b7c8ec4a, 1623e5981b3249a590e22f43fe5d3aae8bef6e47, 3e66abac0ec5f0c0c94c12d5e541443b7a499d3d, 49d9c2ee84d5b069e9e5c3bfada3445debc7979b, 5420737e851df9a72022bf4bca3b5d649a9191bb, a4ea9fb4804e82475fc0fdacd7bcd7240a86457d, cc2c397d56350e5c108bf75eb6a3a08806d89a75, f9c2e2d83575101e92e8a8b9961dca97dda45328, snapshot-233d1f8b1e36823015c41d26b0916b095301614a17c0fca7fc87f30301ef20e0, snapshot-809a8658d827d180d5bc245359932e4a5224d2a6d0028a56707dcbadda3f4a46
- checkpoint hashes observed: 26 (1d6a60c3844d7c1eee23cb7556ce66575081de2475455731c57deaac9f27dd6b; 28924728be3618eb1adfb9c0ae9180afa72c62b825e79329c070e0256595722f; 29e0a54f0045878b11bc8b2b3328b22a5363cb86b45181ac6a678e6ec68e29b9; 39f32c6a61160093cec599c6a1431ac64471c233d50dfe58def8d4f46e5d46d6; 3c74f192e78a9df75f6f1524bb98205616b2e5b8dbe6e489abf936883910dea3; 3ce3f3ae2e50291869f56ef4c930e98963c47f5e54ceb5aaa02f41331c1f879b; 3e0d18b906a02cf7334a4c27296cb232f9f6f576c3df55aa0f34aecf52b7b487; 3f62afde0d4506aa335812b38e083e7b5e94dffeb17302392c3cfa5cd66b1d4d; 56cdbdc6db1fc1e4f0789e97b88235c7ea48dd3d2d2a9883fd48ea14a13165d3; 61cf2151fd012bc1469eafac16597f88052682b969a8690bf4f5c4f628785ba1; 75afcf61badc12221e3a1ea6f9bb0df5d461d258ae437a35c2900d297ff2a313; 899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c; 94c60e9dac6084bdb476f3dc2bfddbd86d0e510b9e619d414840c0054aa0741b; 9ea064dcb16f7d455191b670046cccdfb69eec0f69633ad6209717e8ac94133a; a608431a76eb5d9d9c331aaea22acb67ecea475b706f8cf5324f4a14fb9016f7; b75767acbc2226ace2cfcf4399f981a32283f7b2670a137bf0dd2bce100feea7; b8a9add8142e29bba4aeabac0b1a1890e564cbfaeff7f4ea4fcb54ea0a35c59e; c90fdcdba23d40351b1a36aee85837fc9cf97a1e9f1fbc82f0b13bdc82d7edfb; cc4445c423292fe0af6f2ee13b628ec00b1e3d51cef10c16009d2f9374d4ba76; d5f05e082b0290cc0c661a7c0a14817e05f42d3cf71c544c5344693fc4818db7; e5424809519c37cf9353589779954b39881940a41f8248153c252b7b7f6473d3; e7bdeb6e1a05097491d198e2f496bb6e4316e9aad3bb5d4dd0b19cccc301051f; ec0efc4d8927e00f5b5337740a8b89afaedff05e537bca7b797674564414b026; efe672817bbdf7b592f98c89b8f86372dfc8448d73e97f464efe2f79b4bbe246; fc93bd9c473a766747bbbbd113b66af0aed26645865a3d701dfdca1f0cc89d5a; fd7e864c1ad4f9165a0a51167d917155eb9eb945eaa50ccd33087ff2398617c6)
- data split hashes observed: 4 (4f4dd239ee109fb62bd314f3b8170e21c10c19b55117ec7063df1af52b8cac9d; 68aac70dbf610d47291152370dcfb5de50f75a391a99ad5ef02bb726ed5d6227; c2932626eb5db21a326d17bea01665ab4e76d3cee87bf95b7d30d8fc2e112017; e051aba0785d9136df5e5ee466fa37ad24417d367854134f14ba34e4794f2e10)
- seeds observed: [0, 1, 2]
- failed/partial/unknown runs: 0
- fixed-checkpoint matrix: 162/162 successful
- executed-product fixed-checkpoint matrix: 147/147 successful
- training matrix: 21/21 successful
- trained-evaluation matrix: 315/315 successful
- case-clustered bootstrap replicates: 2000

## H1–H7 verdict

| hypothesis | verdict | evidence_files | caveat |
| --- | --- | --- | --- |
| H1 Local auxiliary-update risk exists | Strongly supported | Figure_R1_source.csv; p0_a_utility_summary.csv | Observed paired inference-time utility; not a causal ground truth. |
| H2 Saved trust signals rank observed utility | Partially supported | p0_b_signal_validity.csv; Figure_R2_trust_validation.png | Ranking and raw calibration are distinct; level-global U limits spatial claims. |
| H3 G adds information beyond U | See measured comparisons | Table_R1_components.csv | A full verdict requires completed three-seed trained variants. |
| H4 Executed product rule outperforms simple alternatives | See measured comparisons | Table_R1_components.csv; Table_R1_product_fixed_corruption.csv | Submitted A_impl, fixed product intervention, and independently trained variants are reported separately. |
| H5 Controlled degradation forms a mechanism-consistent chain | See measured intervention slopes | Figure_R3_source.csv; p0_c_degradation_slopes.csv | Corruption severities have different physical/statistical units and are not pooled numerically. |
| H6 Gains are not explained by capacity or global attenuation | Not evaluated | Table_R1_components.csv; Table_R2_efficiency.csv | Parameter-matched and magnitude-matched controls were not in the minimum 12-hour matrix. |
| H7 Failure boundaries remain measurable | Strongly supported | p0_g_quadrants.csv; predeclared_case_selection.csv | Common-mode failures require a dedicated intervention and are not inferred from other corruptions. |

## Primary findings

### Finding 1: problem existence

See `06_summary_tables/p0_a_utility_summary.csv` and Figure R1. The table preserves mean utility, its full negative tail, harmful-update rate, vessel-group strata, case counts, and case-clustered intervals.

### Finding 2: U validity and limits

See `06_summary_tables/p0_b_signal_validity.csv`. `1-U`, `G`, manuscript-formula `T_formula=G(1-U)`, executed acceptance `A_impl`, and the preregistered FDD orientation are reported with AUPRC, AUROC, Brier, NLL, AURC, prevalence, and bootstrap intervals. Historical raw `T` is never treated as formula trust. Missing confidence/entropy is explicit rather than imputed.

### Finding 3: G incremental role

See Table R1. Fixed-checkpoint interventions are separated from independently trained variants so that output intervention effects are not conflated with training adaptation.

### Finding 4: executed full interpolation vs alternatives

See Table R1, `Table_R1_product_fixed_corruption.csv`, and `p0_f_har_brr.csv`. The submitted `full` row always denotes executed `A_impl=1-(1-G)U`; `product` denotes a separate run that actually executes `G(1-U)`. Both performance and action-quality trade-offs are required before preferring a variant.

### Finding 5: controlled intervention chain

See Figure R3 and `p0_c_degradation_slopes.csv`. Each corruption family retains its original severity unit; the plot normalization is display-only.

### Finding 6: capacity/regularization controls

The original minimum matrix covers no auxiliary, residual, U-only, G-only, additive, and submitted-full controls. The supplemental matrix adds product-gated fixed-checkpoint runs plus three independently trained product seeds. Parameter-matched and global-attenuation controls remain outside the completed package unless separately listed.

### Finding 7: failure boundaries

See Figure R4 and `08_failure_cases/predeclared_case_selection.csv`. Incorrect acceptance and over-suppression remain visible; a common-mode representative is not claimed without a common-mode run.

## Reviewer mapping

### pCat

Use source-to-external signal rows, controlled corruptions, confidence/entropy controls, and Table R2 to address proxy stability, shift behavior, and cost.

### Rkxt

Use paired update utility, explicit `T_formula`/`A_impl` comparisons, fixed and retrained alternatives, HAR/BRR, and Figure R4 to address whether TCFF is more than a renamed gate and where it fails.

### mDyq

Use the mild-to-severe corruption curves, component table, three-seed coverage, and the same-device efficiency benchmark for the requested engineering validation.

### Area Chair

G0 reproduction, multi-seed component retraining, controlled robustness, reliability/action quality, simple alternatives, and efficiency directly address robustness, contribution, and computational cost.

## Main tables and figures

- Figure R1: utility distribution and harmful-update rate.
- Figure R2: formula-trust and executed-acceptance AUPRC/AURC, reliability, and U×G phase diagram.
- Figure R3: controlled corruption mechanism chain.
- Table R1: fixed interventions and independently trained variants.
- Table R2: efficiency and estimator/reliability metrics when benchmark output is present.
- Figure R4: action quadrants and preregistered case selection.

## Negative or mixed findings

- 68/126 primary component clean/pooled-corruption signal rows did not exceed the prevalence AUPRC baseline.
- Level-specific formula-T decisions on clean plus pooled-corruption strata include 19675 incorrect-acceptance rows and 19241 over-suppression rows.

## Recommended claim scope

Describe the results as paired inference-time auxiliary-update utility and within-model response under controlled interventions. Do not call the utility a causal ground truth, do not generalize beyond the measured cohorts/corruptions, and do not claim that TCFF wins every metric or every failure regime.

## Exact numbers safe to quote

The authoritative ledger contains 744 values: `06_summary_tables/exact_numbers_safe_to_quote.csv`. Every quoted number must pass `tcff_verify_quote_table.py` after the final refresh.

## Unresolved issues

- A parameter-matched ordinary gate and dedicated common-mode corruption are not part of the completed minimum package unless separately listed as successful runs.
- Data-integrity overlap evidence must remain machine-readable in DATA_INTEGRITY_AUDIT.json.
