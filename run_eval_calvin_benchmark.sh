#! /bin/bash

# Start the policy server for CALVIN evaluation (run in the streampi conda env).
# Adjust --policy.dir to the checkpoint you want to evaluate.

CUDA_VISIBLE_DEVICES=1 python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_calvin_stream5 \
  --policy.dir=checkpoints/pi05_calvin_stream5/pi05_stream5_calvin_4nodes/29999
