# TCFF rebuttal post-processing protocol

Frozen before inspecting the formal Lausanne or IXI-HH mechanism results.

## Analysis populations

- Primary fixed-checkpoint mechanism analysis: `model_variant=full` using the submitted TCFF seed-42 checkpoint.
- Clean reference: `corruption_type=none`, one deterministic realization per case.
- Corruption analysis: three fixed realizations per non-zero severity. The clean row is reused only as the paired severity-zero anchor and is not counted as three independent observations.
- Trained-component analysis: independently trained `full`, `u_only`, `g_only`, and `residual` variants, seeds 0/1/2. Source validation selects each run's checkpoint; external cohorts are evaluation-only.

## Unit of analysis and de-duplication

- Statistical resampling unit is the case, never a voxel, patch, ROI, or feature-level row.
- Component utility uses exactly one `vessel_component_roi` row per case, vessel class, and feature level. Because the output-level component utility is repeated across feature levels, problem-existence summaries first collapse to one row per case and vessel class.
- Centerline utility uses `centerline_segment_roi`; rows without a valid centerline target remain missing and are excluded only from that target's denominator.
- Voxel utility uses `vessel_component_roi` so that the evaluation region is anatomically interpretable.
- Trust-ranking analyses retain feature level because `U`, `G`, and `T` are level-specific. They never treat repeated levels as independent cases in confidence intervals.

## Signal orientation

Higher scores always mean “more likely beneficial” for ranking and risk–coverage:

- evidence score: `1-U`;
- spatial compatibility score: `G`;
- executed trust score: `T`;
- distance baseline: `1-FDD/2`, clipped to `[0,1]`, using the implementation's theoretical FDD range `[0,2]`.

For harmful-update detection, the complementary orientation is used. AUPRC is reported with the positive-class prevalence and target direction.

## Primary utility target

- Primary: `B_component = 1[Q_component > 0]` on component ROIs.
- Secondary: `B_vox` and `B_centerline` with their matching utility values.
- Exact zero utility is classified as non-beneficial, as specified by `B = 1[Q > 0]`.

## Threshold rule

- Thresholds are chosen only on clean TopCoW validation rows.
- A separate threshold is frozen for each signal, utility target, and feature level.
- Candidate thresholds are the unique source scores plus boundary candidates.
- Selection minimizes balanced action error `(HAR + BRR) / 2`, where `HAR=P(score>tau|B=0)` and `BRR=P(score<=tau|B=1)`.
- Ties are resolved by lower total action error, then higher accepted coverage, then the smaller threshold.
- The frozen thresholds are applied unchanged to Lausanne, IXI-HH, and corrupted runs.

## Uncertainty and multiplicity

- Use 2,000 deterministic case-clustered bootstrap replicates with seed 20260729.
- Report percentile 95% confidence intervals.
- Pair model comparisons by case within cohort and seed where available.
- Holm correction is applied within each declared family of inferential comparisons.
- Descriptive ROI counts are reported, but they are not the inferential sample size.

## Risk–coverage and calibration

- Risk–coverage accepts updates from highest to lowest beneficial score; harmful risk is the fraction with `B=0` among accepted rows.
- Tied scores enter coverage together. AURC is trapezoidal area over the resulting curve, including `(coverage=0,risk=0)`.
- Calibration uses fixed probability mappings above and fixed-width deciles. Each bin reports ROI-row count and distinct case count.
- No external cohort is used to choose a threshold, calibration mapping, severity, model, checkpoint, ROI, or plotting subset.

## Missing signals and negative results

- Confidence and entropy are marked unavailable for runs that did not persist probability summaries; they are never imputed from hard predictions.
- Non-finite or undefined metrics remain missing, with denominators reported.
- Failed and partial runs remain in the audit inventory and are excluded from quantitative pooling unless explicitly labeled.
- No case, severity, corruption family, or metric is removed because its result contradicts a hypothesis.

## Predeclared case selection

Cases are selected mechanically after thresholds are frozen: top five trust–utility disagreement, highest HAR, highest BRR, largest Det-F1 improvement, largest Det-F1 degradation, and representative common-mode failure if such a run exists. Ties use lexical case ID.
