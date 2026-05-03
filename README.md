# clarify

Research code and final report for **intent stabilization for ambiguous user requests via all-stage resampling**.

The project studies an inference-time wrapper that samples structured ambiguity, intent, and response-strategy hypotheses before answering. When the sampled state is unstable, the wrapper resamples the full stage bundle and asks a clarification question if uncertainty remains high.

## Setup

Python `3.11` is the supported runtime.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
./.venv/bin/pip install -e ".[dev]"
export TINKER_API_KEY=...
```

You can also place the key in `.env`; the runtime loads it for long-running jobs.

## Data

Frozen data artifacts live under `data/`.

```bash
./.venv/bin/python scripts/prepare_data.py
./.venv/bin/python scripts/validate_data.py \
  --manifest data/infoquest_manifest.json \
  --manifest data/clarifybench_v1_manifest.json
```

## Run A Method

```bash
./.venv/bin/python scripts/run_inference.py \
  --method resample_clarify \
  --dataset infoquest \
  --split test \
  --limit 8
```

## Sensitivity Runs

```bash
ROOT=outputs/k_sensitivity_$(date +%Y%m%d_%H%M%S)

./.venv/bin/python scripts/run_k_sensitivity.py \
  --config configs/report_baselines.yaml \
  --output-dir "$ROOT" \
  --dataset infoquest \
  --split dev \
  --model-label qwen3_30b \
  --k 1 --k 3 --k 5 --k 7
```
