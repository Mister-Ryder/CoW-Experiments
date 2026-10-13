# Result Schema Notes

- `Q_*` is observed auxiliary-update utility under the fixed within-model intervention: unconditional residual metric minus no-aux metric. It is not called causal ground truth or true reliability.
- `B_* = 1` means the corresponding observed utility is positive; zero means non-positive.
- `Q_target_*` is the selected target variant's metric change relative to no-aux.
- `U`, `G`, `T`, `FDD`, `delta_l1`, and `accepted_delta_l1` are reduced within the mapped ROI at one feature level.
- `T` is the executed effective accepted fraction. The locked full path exports `T=1-wU`, where `w` is the learned spatial suppression factor.
- `trust_bin` uses fixed deciles and is not a tuned threshold.
- `utility_quadrant_at_0p5` is a descriptive fixed-threshold pilot field. Formal HAR/BRR thresholds will be selected only on the source-development set and applied unchanged to external sets.
- Tiny output-space masks can vanish under nearest mapping to a coarse feature grid. Such signal fields are NA and excluded pairwise; they are never imputed as zero.
- HD95 is in preprocessed-voxel units under the historical evaluator.
- Formal mechanism inference is single-view/no-TTA so that one unambiguous internal signal realization maps to each prediction. Separate G0 reproduction uses the historical four-view TTA protocol.

