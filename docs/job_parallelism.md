# Parallel And Resumable Execution

The expensive phases are designed to be split across terminals.

## Command Generator

Use:

```bash
./.venv/bin/python scripts/plan_jobs.py \
  --config configs/report_baselines.yaml \
  --phase all \
  --output-root outputs/research_pipeline
```

This prints terminal-safe commands for:

- dev sweep
- sharded baseline leaves
- aggregation and repeatability
- resample ablations
- audit export
- teacher rollout shards
- student SFT
- student eval
- optional DPO preparation

## Recommended Terminal Allocation

For `8` terminals:

- terminal 1: dev sweep or baseline aggregation / repeatability
- terminals 2-4: sharded baseline leaves
- terminals 5-6: teacher rollout shards
- terminal 7: ablations
- terminal 8: audit export, paper tables, and local post-processing

For `4` terminals:

- terminal 1: dev sweep, then aggregation
- terminals 2-3: baseline shards
- terminal 4: ablations or teacher rollouts

## Resume Strategy

- baseline leaves resume at the shard and method level
- teacher rollout export resumes at the shard and rollout level
- the research pipeline script can skip completed phases with `--skip-*`
- paper generation is cheap and can be rerun whenever artifacts change
- for unstable network conditions, wrap long jobs with `scripts/retry_forever.py` so a crashed process restarts and resumes from disk automatically
- Tinker telemetry is disabled by default and subprocess sampling is disabled by default in the runtime config; after a sleep/resume event, the preferred recovery path is to fail fast and let `retry_forever.py` restart a fresh process

Example:

```bash
./.venv/bin/python scripts/retry_forever.py --retry-delay 30 -- \
  ./.venv/bin/python scripts/run_sharded_leaf.py ...
```

## Common Pattern

1. Run `scripts/plan_jobs.py`.
2. Paste the shard commands into multiple terminals.
3. Wait for all shards of a leaf to finish.
4. Run the aggregate-only command if applicable.
5. Refresh the matrix or rollout root.
6. Move on to the next phase.

## Manifest Utility

Use this helper to read the trained checkpoint path without `jq`:

```bash
./.venv/bin/python scripts/read_manifest_value.py \
  --path outputs/student_sft/student_sft_manifest.json \
  --key checkpoint_path
```
