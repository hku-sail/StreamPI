import dataclasses

import einops
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model


def make_calvin_example() -> dict:
    """Creates a random input example for the calvin policy."""
    return {
        "state_ee_pos": np.random.rand(3),
        "state_ee_rot": np.random.rand(3),
        "state_gripper": np.random.rand(1),
        "image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif image.ndim == 4 and image.shape[1] == 3:
        # History-stacked frames: (t, c, h, w) -> (t, h, w, c).
        image = einops.rearrange(image, "t c h w -> t h w c")
    return image


@dataclasses.dataclass(frozen=True)
class CalvinInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # History-stacked frames arrive as (t, C, H, W) and are converted to (t, H, W, C).
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            # CALVIN state: ee position (3) + ee rotation (3) + gripper (1) = 7-dim.
            "state": np.concatenate(
                [
                    data["state_ee_pos"],
                    data["state_ee_rot"],
                    data["state_gripper"],
                ],
                axis=0,
            ),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training. CALVIN actions are already delta actions:
        # delta ee position (3) + delta ee rotation (3) + gripper (1) = 7-dim.
        if "action_delta_ee_pos" in data and "action_delta_ee_rot" in data and "action_gripper" in data:
            inputs["actions"] = torch.cat(
                [data["action_delta_ee_pos"], data["action_delta_ee_rot"], data["action_gripper"]], axis=1
            )

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CalvinOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For calvin, we only return the first 7 actions (since the rest is padding).
        return {"actions": np.asarray(data["actions"][:, :7])}
