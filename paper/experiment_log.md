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

## 2026-04-21

- Completed enough of the InfoQuest dev sweep to identify the first winning resampling configuration: `k3_r1_c08`.
- Verified the current dev result: `resample_clarify` TSR `0.29` vs `targeted_clarify` TSR `0.24`, with equal `1.00` appropriate-action rate.
- Updated the report and HTML spotlight deck so that all unfinished Phase 1 and Phase 2 artifacts render as explicit placeholders rather than silently reusing legacy clarification-only outputs.

## 2026-04-22

- Completed the fresh `InfoQuest test / qwen3_30b` gate leaf and found that the first `resample_clarify` controller underperformed: `0.22` TSR vs `0.26` for `targeted_clarify` and `0.275` for `generic_clarify`.
- Diagnosed the failure mode from the gate metrics and in-progress ablations: the initial controller was over-resampling, almost always still asking for clarification, and not converting that extra compute into better outcomes.
- Revised the main method to use a full repair pass by default, generic clarification as the default fallback, and calibration primarily for reporting rather than action selection.
- Left the current ablation run alive as comparison evidence and prepared a clean fresh-output rerun path for the simplified controller.

## How To Use This Log

- Record the chosen dev-sweep configuration and calibrator path before frozen test runs.
- Record failed runs with a short cause and affected models.
- Record the final baseline matrix root, ablation root, repeatability outputs, teacher-rollout root, student training outputs, and audit agreement artifact paths once frozen.
