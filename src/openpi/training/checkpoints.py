from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import pathlib
import shutil
import time
from typing import Protocol

from etils import epath
import jax
import jax.experimental.multihost_utils as multihost_utils
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


_CHECKPOINT_SUCCESS_FILE = "_SUCCESS"


def _success_marker(step_dir: pathlib.Path) -> pathlib.Path:
    return step_dir / _CHECKPOINT_SUCCESS_FILE


def _wait_for_success_marker(step_dir: pathlib.Path) -> None:
    marker = _success_marker(step_dir)
    logging.info(f"[rank {jax.process_index()}] Waiting for checkpoint marker {marker}")
    while not marker.exists():
        time.sleep(5)
    logging.info(f"[rank {jax.process_index()}] Observed checkpoint marker {marker}")


def _is_checkpoint_step_dir(path: pathlib.Path) -> bool:
    if not path.is_dir() or not path.name.isdigit():
        return False
    if _success_marker(path).exists():
        return True
    # Backward compatibility for legacy Orbax checkpoints written before the
    # success marker was introduced.
    return (path / "params").exists() and (path / "train_state").exists()


def checkpoint_steps(checkpoint_dir: epath.Path | pathlib.Path | str) -> list[int]:
    checkpoint_dir = pathlib.Path(str(checkpoint_dir))
    if not checkpoint_dir.exists():
        return []
    return sorted(int(d.name) for d in checkpoint_dir.iterdir() if _is_checkpoint_step_dir(d))


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    # if checkpoint_dir.exists():
    #     if overwrite:
    #         checkpoint_dir.rmtree()
    #         checkpoint_dir.mkdir(parents=True, exist_ok=True)
    #         logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
    #     elif resume:
    #         resuming = True
    #     else:
    #         raise FileExistsError(
    #             f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
    #             "to indicate how to handle it."
    #         )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if resume is True:
        resuming = True

    if jax.process_index() == 0:
        if checkpoint_dir.exists():
            if overwrite:
                checkpoint_dir.rmtree()
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
            elif resume:
                logging.info(f"Resuming training from checkpoint directory {checkpoint_dir}")
                resuming = True
            else:
                raise FileExistsError(
                    f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                    "to indicate how to handle it."
                )
                pass

    # Clean up any leftover Orbax temp directories from previous failed saves.
    if jax.process_index() == 0:
        for tmp_dir in checkpoint_dir.glob("*.orbax-checkpoint-tmp-*"):
            if tmp_dir.is_dir():
                logging.warning(f"Removing leftover temp checkpoint dir: {tmp_dir}")
                tmp_dir.rmtree()

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=True,
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)
    checkpoint_manager.wait_until_finished()

    # Save assets directly after checkpoint is written (same as save_state_multihost).
    if jax.process_index() == 0:
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            step_dir = pathlib.Path(str(checkpoint_manager.directory)) / str(step)
            assets_save_path = step_dir / "assets" / data_config.asset_id
            try:
                _normalize.save(assets_save_path, norm_stats)
                logging.info(f"Saved assets to {assets_save_path}")
            except Exception as e:
                logging.error(f"Failed to save assets: {e}")
        else:
            logging.warning(f"Skipping assets save: norm_stats={'None' if norm_stats is None else 'ok'}, asset_id={data_config.asset_id}")


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def __init__(self):
        self._executor = futures.ThreadPoolExecutor(max_workers=1)

    def close(self):
        self._executor.shutdown()

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [self._executor.submit(self.save, directory, args)]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])


# ============================================================
# Multi-host checkpoint utilities
# ============================================================


def _cleanup_old_checkpoints(
    checkpoint_dir: epath.Path, max_to_keep: int, keep_period: int | None
):
    """Remove old checkpoint directories, respecting max_to_keep and keep_period."""
    checkpoint_dir = epath.Path(str(checkpoint_dir))
    all_steps = checkpoint_steps(checkpoint_dir)
    steps_to_keep = set(all_steps[-max_to_keep:])
    if keep_period:
        steps_to_keep.update(s for s in all_steps if s % keep_period == 0)
    for s in all_steps:
        if s not in steps_to_keep:
            (checkpoint_dir / str(s)).rmtree()


def save_state_multihost(
    checkpoint_dir: epath.Path,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    max_to_keep: int = 1,
    keep_period: int | None = None,
):
    """Save a checkpoint using Orbax's native multi-process JAX array path.

    All processes must call this function. We intentionally keep sharded JAX
    arrays sharded and let Orbax coordinate the write; converting to NumPy first
    makes non-primary ranks finish too early and leaves rank 0 alone in a large
    filesystem write.
    """
    with at.disable_typechecking():
        train_state, params = _split_params(state)

    jax.block_until_ready((train_state, params))

    logging.info(f"[rank {jax.process_index()}] Saving multihost checkpoint at step {step}")

    step_dir = pathlib.Path(str(checkpoint_dir)) / str(step)
    if jax.process_index() == 0:
        if step_dir.exists():
            logging.warning(f"[rank 0] Removing existing checkpoint directory before save: {step_dir}")
            shutil.rmtree(step_dir)
        step_dir.mkdir(parents=True, exist_ok=True)

    multihost_utils.sync_global_devices(f"save_state_multihost_prepare_{step}")

    with ocp.PyTreeCheckpointer(use_ocdbt=True) as ckptr:
        ckptr.save(
            epath.Path(str(step_dir / "params")),
            ocp.args.PyTreeSave({"params": params}),
        )

    if jax.process_index() == 0:
        logging.info(f"[rank 0] Saved params checkpoint to {step_dir / 'params'}")

    with ocp.PyTreeCheckpointer(use_ocdbt=True) as ckptr:
        ckptr.save(
            epath.Path(str(step_dir / "train_state")),
            ocp.args.PyTreeSave(train_state),
        )

    if jax.process_index() == 0:
        logging.info(f"[rank 0] Saved train_state checkpoint to {step_dir / 'train_state'}")

        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            assets_save_path = step_dir / "assets" / data_config.asset_id
            try:
                _normalize.save(assets_save_path, norm_stats)
                logging.info(f"[rank 0] Saved assets to {assets_save_path}")
            except Exception as e:
                logging.error(f"[rank 0] Failed to save assets: {e}")

        _cleanup_old_checkpoints(checkpoint_dir, max_to_keep, keep_period)

        _success_marker(step_dir).write_text(f"step={step}\n")
    else:
        _wait_for_success_marker(step_dir)

    logging.info(f"[rank {jax.process_index()}] Checkpoint saved at step {step}")


def restore_state_multihost(
    checkpoint_dir: epath.Path,
    state_sharding,
    step: int | None = None,
) -> training_utils.TrainState:
    """Restore from a multihost checkpoint and re-shard for multi-node training.

    All processes call this. Each reads the full checkpoint then reshards locally.
    Supports checkpoints saved by the native multi-process Orbax path and legacy Orbax checkpoints.
    """
    all_steps = checkpoint_steps(checkpoint_dir)
    if not all_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    if step is None:
        step = all_steps[-1]

    step_dir = pathlib.Path(str(checkpoint_dir)) / str(step)
    logging.info(f"Restoring multihost checkpoint from step {step}: {step_dir}")

    with at.disable_typechecking():
        train_state_sharding, params_sharding = _split_params(state_sharding)

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(epath.Path(str(step_dir / "params")))
        params = ckptr.restore(
            epath.Path(str(step_dir / "params")),
            ocp.args.PyTreeRestore(
                item={"params": metadata["params"]},
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=None, restore_type=np.ndarray),
                    {"params": metadata["params"]},
                ),
            ),
        )
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(epath.Path(str(step_dir / "train_state")))
        train_state = ckptr.restore(
            epath.Path(str(step_dir / "train_state")),
            ocp.args.PyTreeRestore(
                item=metadata,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=None, restore_type=np.ndarray),
                    metadata,
                ),
            ),
        )

    # Re-shard onto the multi-host mesh.
    train_state = jax.device_put(train_state, train_state_sharding)
    params = jax.device_put(params, {"params": params_sharding})

    return _merge_params(train_state, params)
