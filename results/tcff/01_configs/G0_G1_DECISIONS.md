# G0 and G1 Decisions

## G0: PASS

The original TCFF source-development, Lausanne, and IXI-HH results and a concat baseline were rerun from immutable checkpoints. Continuous metric differences are numerical-scale; Det-F1 is exact. See `00_project_discovery/BASELINE_REPRODUCTION.md`.

## G1: PASS

### Non-invasive instrumentation

- Pre-instrumentation full-output SHA-256: `f1142e5b8ce7d78d90ce5d0222478b5b553daa761441fa065eaf594c8a689aea`.
- Post-instrumentation default-path SHA-256: identical.
- Instrumentation-enabled clean-path SHA-256: identical.
- Maximum absolute probability difference: 0.0.
- All exported tensors finite.

Exported signal grids are shallow 64×160×160, mid 32×80×80, and deep 16×40×40. U is level-global and broadcast; G/T/FDD are spatial; channel weights are per channel; Δ and accepted Δ preserve feature channels.

### Interventions and corruptions

The one-case G1 test executed and repeated:

- Gaussian feature noise.
- Feature translation.
- Channel dropout.
- Feature attenuation.
- Cyclic donor semantic mixing.
- Full, U-only, G-only, unconditional residual, no-aux, and additive interventions.
- Gate dose alpha = 0, 0.25, 0.5, 0.75, 1.

Every same-seed corruption repeated with maximum absolute output difference 0.0. Each nonzero corruption changed the clean floating output. Alpha 0 matched no-aux and alpha 1 matched unconditional residual within the 1e-6 acceptance tolerance.

### Result interface

The first complete runner case succeeded with:

- one case row;
- 13 class rows;
- 108 internal/region rows;
- component, centerline, and fixed-patch ROIs;
- all three feature levels;
- manifest, config hash, checkpoint hash, split hash, status, runtime, CSV, and Parquet outputs.

Six internal fields are NA because tiny masks did not occupy a voxel on the deep grid. This is the specified missing-value behaviour and is documented in `SCHEMA_NOTES.md`.

### Training entry

An audited U-only one-epoch run completed in 112.8 seconds including source-development evaluation and final TTA, wrote a selected checkpoint, and closed its manifest with `status=success`. The measured training epoch itself took approximately 72 seconds.

### Preserved pre-run failures

Two engineering failures occurred before model/data loading: a missing fixed log directory and a system `libstdc++` precedence conflict. Both failure records/logs are retained in `11_quality_audit`; the fixes were directory creation and pinning the existing `cu122py39` runtime library path. Neither attempt contributed a metric row.

Formal P0 execution was authorized only after all checks above passed.

