"""Compute norm stats for pi05_calvin_stream5.

`scripts/compute_norm_stats.py` writes to `assets_dirs / repo_id`, which lands inside the
dataset directory when repo_id is an absolute local path. This wrapper reuses the same
dataloader but saves to the assets directory that the config actually loads from:
`assets/pi05_calvin/InternRobotics/InternData-Calvin_ABC/norm_stats.json`.
"""

import pathlib

import numpy as np
import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as _config
from scripts.compute_norm_stats import create_torch_dataloader

CONFIG_NAME = "pi05_calvin_stream5"
NUM_WORKERS = 32
OUTPUT_PATH = pathlib.Path("assets/pi05_calvin/InternRobotics/InternData-Calvin_ABC")


def main() -> None:
    config = _config.get_config(CONFIG_NAME)
    data_config = config.data.create(config.assets_dirs, config.model)
    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, config.model, NUM_WORKERS
    )
    print(f"num_batches: {num_batches}", flush=True)

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats[key].get_statistics() for key in keys}
    print(f"Writing stats to: {OUTPUT_PATH}")
    normalize.save(OUTPUT_PATH, norm_stats)
    print("done")


if __name__ == "__main__":
    main()
