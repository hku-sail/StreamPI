# CALVIN Benchmark

This directory contains the client-side evaluation entry point for the [CALVIN benchmark](https://github.com/mees/calvin),
adapted for the stream-style `pi05_calvin_stream5` training config in this repo.

For policy training, use the LeRobot dataset `InternData-Calvin_ABC` with:

- [`CalvinInputs` and `CalvinOutputs`](../../src/openpi/policies/calvin_policy.py)
- [`LeRobotCalvinDataConfig`](../../src/openpi/training/config.py)
- [`pi05_calvin_stream5`](../../src/openpi/training/config.py) (train with `train_stream5_calvin.sh`)

## Setup

Clone the CALVIN repo:

```bash
git clone https://github.com/mees/calvin.git /path/to/calvin
```

Create a conda environment (e.g. `calvin_venv`) and install dependencies following the
[official instructions](https://github.com/mees/calvin#computer--quick-start), then install the client package
from this repo root:

```bash
cd /home/ma-user/work/users/jhhou/projects/StreamPI_RoboV2-main
pip install -e packages/openpi-client
pip install draccus
```

Make sure you have downloaded the CALVIN dataset so that `dataset/task_ABC_D/validation/` exists under the
CALVIN repo root.

## Evaluation

Evaluation uses two processes:

1. A policy server running in the main `streampi` environment.
2. A CALVIN client running in the `calvin_venv` environment.

### Start the policy server

```bash
bash run_eval_calvin_benchmark.sh
```

or directly:

```bash
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_calvin_stream5 \
  --policy.dir=checkpoints/pi05_calvin_stream5/pi05_stream5_calvin/29999 \
  --port 8000
```

### Run evaluation

```bash
conda activate calvin_venv

export CALVIN_ROOT=/path/to/calvin
bash run_calvin_env.sh
```

or directly:

```bash
export PYTHONPATH=$PYTHONPATH:$PWD:$CALVIN_ROOT

python examples/calvin/main.py \
  --calvin_root /path/to/calvin \
  --save_name pi05_calvin_stream5 \
  --host 0.0.0.0 \
  --port 8000
```

Useful options (see [main.py](main.py) for the full list):

- `--replan_steps`: Number of actions executed from each predicted chunk before re-planning (default 5,
  matching `hist_interval=5` of the stream training config).
- `--num_sequences`: Number of evaluation sequences (default 1000, standard ABC->D protocol).
- `--ep_len`: Max environment steps per instruction (default 720).
- `--debug`: Save rollout GIFs and print per-subtask results.
- `--out_path` / `--save_name`: Logs are written under `{out_path}/{save_name}/`.

Results are written to:

```text
{out_path}/{save_name}/result.json
{out_path}/{save_name}/success_rate.txt
```

`result.json` contains:

- `avg_seq_len`: average successful sequence length in each 5-instruction chain.
- `chain_sr`: success rates for completing 1, 2, 3, 4, and 5 instructions in a row.
- `task_info`: per-task success and total counts.

## Notes on the streaming client

Unlike the upstream FASTER CALVIN client, this client sends a `"step"` counter with every inference call
(it restarts at 0 for every evaluation sequence). The stream model on the server resets its KV-cache memory
whenever `step % hist_horizon == 0` (hist_horizon=5 for `pi05_calvin_stream5`), so episode boundaries and
history re-anchoring are handled automatically as long as the counter keeps incrementing per inference call.
