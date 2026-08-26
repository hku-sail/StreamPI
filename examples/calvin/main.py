import copy
import json
import logging
import os
import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import draccus
import hydra
import numpy as np
import torch
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import count_success, get_env_state_for_initial_condition, get_log_dir
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from termcolor import colored
from tqdm.auto import tqdm

from examples.calvin.calvin_env_wrapper import CalvinEnvWrapperRaw

logger = logging.getLogger(__name__)

os.environ["FFMPEG_BINARY"] = "auto-detect"


class ClientModel(CalvinBaseModel):
    def __init__(self, host, port, replan_steps):
        super().__init__()
        self.client = _websocket_client_policy.WebsocketClientPolicy(host, port)
        self.replan_steps = replan_steps
        # Inference-call counter within the current sequence. Sent to the server as "step";
        # the stream model resets its KV-cache memory whenever step % hist_horizon == 0,
        # so the counter must restart at 0 for every new evaluation sequence.
        self.counter = 0

    def reset(self):
        self.counter = 0

    def step(self, obs, instruction):
        img = obs["rgb_obs"]["rgb_static"]
        wrist_img = obs["rgb_obs"]["rgb_gripper"]

        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))

        element = {
            "image": img,
            "wrist_image": wrist_img,
            "prompt": instruction,
            "state_ee_pos": obs["robot_obs"][:3],
            "state_ee_rot": obs["robot_obs"][3:6],
            "state_gripper": obs["robot_obs"][-1:],
            "step": self.counter,
        }
        self.counter += 1

        action_chunk = self.client.infer(element)["actions"].copy()
        assert len(action_chunk) >= self.replan_steps

        # Only execute the first `replan_steps` actions of the chunk (matches the
        # stream training setup where history frames are spaced by `replan_steps`).
        return action_chunk[: self.replan_steps]


def set_seed_everywhere(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def print_and_save(results, sequences, eval_result_path):
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

    with open(os.path.join(eval_result_path, "result.json"), "w") as file:
        json.dump(data, file)


def make_env(dataset_path, observation_space, device):
    val_folder = Path(dataset_path) / "validation"
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def evaluate_policy(
    model,
    env,
    calvin_root,
    eval_dir,
    ep_len,
    num_sequences,
    debug=False,
):
    conf_dir = Path(calvin_root) / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(num_sequences)

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len
        )  # number of success tasks
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with open(os.path.join(eval_dir, "success_rate.txt"), "a") as f:
                line = f"{sequence_i}/{num_sequences}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)])
                + f" {np.mean(results):.3f} |"
            )
        sequence_i += 1
    print_and_save(results, eval_sequences, eval_dir)
    return results


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    model.reset()
    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(
            env, model, task_checker, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len
        )
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    start_info = env.get_info()
    if debug:
        img_dict = {
            "static": [],
            "gripper": [],
        }

    action_queue = deque()
    for _ in range(ep_len):

        # get action chunk
        if len(action_queue) == 0:
            action_queue.extend(model.step(obs, lang_annotation))

        action = action_queue.popleft()
        if action[-1] < 0:
            action[-1] = -1
        else:
            action[-1] = 1
        obs, _, _, current_info = env.step(action)

        if debug:
            img_dict["static"].append(copy.deepcopy(obs["rgb_obs"]["rgb_static"]))
            img_dict["gripper"].append(copy.deepcopy(obs["rgb_obs"]["rgb_gripper"]))

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                save_debug_gifs(img_dict, eval_dir, sequence_i, subtask_i, subtask, "succ")
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        save_debug_gifs(img_dict, eval_dir, sequence_i, subtask_i, subtask, "fail")
    return False


def save_debug_gifs(img_dict, eval_dir, sequence_i, subtask_i, subtask, status):
    for key, frames in img_dict.items():
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_gif(os.path.join(eval_dir, f"{sequence_i}-{subtask_i}-{subtask}-{key}-{status}.gif"), fps=30)


@dataclass
class GenerateConfig:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000

    #################################################################################################################
    # CALVIN environment parameters
    #################################################################################################################
    # Path to the root of the CALVIN repo (containing calvin_env/, calvin_agent/, calvin_models/, dataset/).
    calvin_root: str = "/path/to/calvin"
    # Number of actions executed from each predicted chunk before re-planning. Should match the
    # hist_interval used during stream training (pi05_calvin_stream5 uses hist_horizon=5, hist_interval=5).
    replan_steps: int = 5
    # Max environment steps per instruction.
    ep_len: int = 720
    # Number of evaluation sequences (1000 = standard ABC->D protocol, 5 chained instructions each).
    num_sequences: int = 1000
    # Save rollout GIFs and print per-subtask results.
    debug: bool = False

    #################################################################################################################
    # Utils
    #################################################################################################################
    out_path: str = "data/calvin_eval"  # Local directory for eval logs
    save_name: str = ""  # Name to save the evaluation results

    seed: int = 0  # Random Seed (for reproducibility)


@draccus.wrap()
def main(cfg: GenerateConfig) -> None:
    # Set seed for reproducibility
    set_seed_everywhere(cfg.seed)

    assert cfg.save_name != "", "Save name is required"

    # Set up paths and environment
    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    eval_dir = os.path.join(cfg.out_path, cfg.save_name)
    os.makedirs(eval_dir, exist_ok=True)

    env = make_env(Path(cfg.calvin_root) / "dataset" / "task_ABC_D", observation_space, torch.device("cuda"))
    model = ClientModel(cfg.host, cfg.port, cfg.replan_steps)

    # Evaluate policy
    evaluate_policy(
        model,
        env,
        calvin_root=cfg.calvin_root,
        eval_dir=eval_dir,
        ep_len=cfg.ep_len,
        num_sequences=cfg.num_sequences,
        debug=cfg.debug,
    )


if __name__ == "__main__":
    main()
