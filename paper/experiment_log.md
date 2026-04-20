# Experiment Log

## 2026-04-14

- Created the baseline-ready manuscript scaffold.
- Switched the report workflow to frozen dataset splits and a separate judge model.
- Added automated table generation from run artifacts.

## 2026-04-20

- Pivoted the repository framing from clarification benchmarking to intent stabilization via selective resampling.
- Added the `resample_clarify` method, disagreement-based stability aggregation, confidence calibration, and resampling-specific metrics.
- Added teacher-rollout export, modular SFT scaffolding, optional DPO preparation, student evaluation, and job-planning entrypoints.
- Preserved the earlier clarification-only results as legacy artifacts for comparison, not as the final paper root.

## How To Use This Log

- Record the chosen dev-sweep configuration and calibrator path before frozen test runs.
- Record failed runs with a short cause and affected models.
- Record the final baseline matrix root, ablation root, repeatability outputs, teacher-rollout root, student training outputs, and audit agreement artifact paths once frozen.
