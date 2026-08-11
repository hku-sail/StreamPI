import dataclasses
import math
from typing import ClassVar

import einops
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing_extensions import Literal

from openpi import transforms


def make_agilex_example() -> dict:
    """Creates a random input example for the Agilex policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


class C_PiperForwardKinematics:
    def __init__(self, dh_is_offset: Literal[0x00, 0x01] = 0x01):
        self.RADIAN = 180 / math.pi
        self.PI = math.pi
        # Denavit-Hartenberg parameters for each link
        # _a: link lengths
        # _alpha: link twists
        # _theta: joint angles
        # _d: link offsets
        self._a = [0, 0, 285.03, -21.98, 0, 0]
        self._alpha = [0, -self.PI / 2, 0, self.PI / 2, -self.PI / 2, self.PI / 2]
        self._theta = [0, -self.PI * 174.22 / 180, -100.78 / 180 * self.PI, 0, 0, 0]
        self._d = [123, 0, 0, 250.75, 0, 91]
        self.init_pos = [55.0, 0.0, 205.0, 0.0, 85.0, 0.0]  # unit xyz-mm, rpy-degree
        # if j2, j3 offset 2°
        if dh_is_offset == 0x01:
            self._a = [0, 0, 285.03, -21.98, 0, 0]
            self._alpha = [0, -self.PI / 2, 0, self.PI / 2, -self.PI / 2, self.PI / 2]
            self._theta = [0, -self.PI * 172.22 / 180, -102.78 / 180 * self.PI, 0, 0, 0]
            self._d = [123, 0, 0, 250.75, 0, 91]
            self.init_pos = [56.128, 0.0, 213.266, 0.0, 85.0, 0.0]  # unit xyz-mm, rpy-degree

    def __MatrixToeula(self, T):
        """
        Convert a transformation matrix to Euler angles (roll, pitch, yaw).
        T: 4x4 transformation matrix
        """
        Pos = [0.0] * 6
        # Extract position (x, y, z)
        Pos[0] = T[3]  # x position
        Pos[1] = T[7]  # y position
        Pos[2] = T[11]  # z position
        # Calculate Euler angles (roll, pitch, yaw) based on rotation matrix
        if T[8] < -1 + 0.0001:
            Pos[4] = self.PI / 2 * self.RADIAN  # pitch (beta)
            Pos[5] = 0
            Pos[3] = math.atan2(T[1], T[5]) * self.RADIAN  # roll (alpha)
        elif T[8] > 1 - 0.0001:
            Pos[4] = -self.PI / 2 * self.RADIAN  # pitch (beta)
            Pos[5] = 0
            Pos[3] = -math.atan2(T[1], T[5]) * self.RADIAN  # roll (alpha)
        else:
            # General case for Euler angles computation
            _bt = math.atan2(-T[8], math.sqrt(T[0] * T[0] + T[4] * T[4]))  # pitch (beta)
            Pos[4] = _bt * self.RADIAN
            Pos[5] = math.atan2(T[4] / math.cos(_bt), T[0] / math.cos(_bt)) * self.RADIAN  # yaw (gamma)
            Pos[3] = math.atan2(T[9] / math.cos(_bt), T[10] / math.cos(_bt)) * self.RADIAN  # roll (alpha)

        return Pos

    def __MatMultiply(self, matrix1, matrix2, m, l, n):
        """
        Multiply two matrices
        matrix1: first matrix
        matrix2: second matrix
        m: number of rows in matrix1
        l: number of columns in matrix1 (rows in matrix2)
        n: number of columns in matrix2
        """
        matrixOut = [0.0] * (m * n)
        for i in range(m):
            for j in range(n):
                tmp = 0.0
                for k in range(l):
                    tmp += matrix1[l * i + k] * matrix2[n * k + j]
                matrixOut[n * i + j] = tmp
        return matrixOut

    def __LinkTransformtion(self, alpha, a, theta, d):
        """
        Compute the transformation matrix for a single link using the Denavit-Hartenberg parameters
        alpha: link twist
        a: link length
        theta: joint angle
        d: link offset
        """
        # Precompute trigonometric functions for efficiency
        calpha = math.cos(alpha)
        salpha = math.sin(alpha)
        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        T = [0.0] * 16  # 4x4 transformation matrix
        T[0] = ctheta
        T[1] = -stheta
        T[2] = 0
        T[3] = a

        T[4] = stheta * calpha
        T[5] = ctheta * calpha
        T[6] = -salpha
        T[7] = -salpha * d

        T[8] = stheta * salpha
        T[9] = ctheta * salpha
        T[10] = calpha
        T[11] = calpha * d

        T[12] = 0
        T[13] = 0
        T[14] = 0
        T[15] = 1

        return T

    def CalFK(self, cur_j):
        """
        Calculate Forward Kinematics for a given joint configuration
        cur_j: list of joint angles
        Returns the positions and Euler angles for each link
        """
        # Initialize transformation matrices
        _Rt = [[0.0] * 16 for _ in range(6)]

        # Compute the individual transformation matrices
        for i in range(6):
            c_theta = cur_j[i] + self._theta[i]
            _Rt[i] = self.__LinkTransformtion(self._alpha[i], self._a[i], c_theta, self._d[i])

        # Multiply transformation matrices
        R02 = self.__MatMultiply(_Rt[0], _Rt[1], 4, 4, 4)
        R03 = self.__MatMultiply(R02, _Rt[2], 4, 4, 4)
        R04 = self.__MatMultiply(R03, _Rt[3], 4, 4, 4)
        R05 = self.__MatMultiply(R04, _Rt[4], 4, 4, 4)
        R06 = self.__MatMultiply(R05, _Rt[5], 4, 4, 4)

        # Extract Euler angles for each transformation
        j_pos = []
        j_pos.append(self.__MatrixToeula(_Rt[0]))  # Euler angles for link1
        j_pos.append(self.__MatrixToeula(R02))  # Euler angles for link2
        j_pos.append(self.__MatrixToeula(R03))  # Euler angles for link3
        j_pos.append(self.__MatrixToeula(R04))  # Euler angles for link4
        j_pos.append(self.__MatrixToeula(R05))  # Euler angles for link5
        j_pos.append(self.__MatrixToeula(R06))  # Euler angles for link6

        return j_pos


def joint_to_ee6d(joint: np.ndarray, binary_gripper: bool = False) -> np.ndarray:
    if joint.shape != (14,):
        raise ValueError(f"Expected joint to have shape (14,), got {joint.shape}")

    # Initialize forward kinematics for both arms
    fk_left = C_PiperForwardKinematics(dh_is_offset=0x01)
    fk_right = C_PiperForwardKinematics(dh_is_offset=0x01)

    # Extract joint angles for each arm
    left_joints = joint[0:6]  # 6 DOF for left arm
    right_joints = joint[7:13]  # 6 DOF for right arm

    # Extract gripper positions
    if binary_gripper:
        left_gripper = joint[6] < 0.02
        right_gripper = joint[13] < 0.02
    else:
        left_gripper = joint[6]
        right_gripper = joint[13]

    # Calculate forward kinematics for both arms
    left_eef_poses = fk_left.CalFK(left_joints)
    right_eef_poses = fk_right.CalFK(right_joints)

    # Get end-effector pose (last link, index 5)
    left_eef_pose = np.array(left_eef_poses[5])  # [x, y, z, roll, pitch, yaw]
    right_eef_pose = np.array(right_eef_poses[5])  # [x, y, z, roll, pitch, yaw]

    # Combine into final output
    eef_pos = np.zeros(20)
    eef_pos[0:3] = left_eef_pose[0:3] / 1000
    left_euler = left_eef_pose[3:6] / 180 * np.pi  # convert roll, pitch, yaw to radians
    left_6d = euler_to_rotate6d(left_euler, "xyz")
    eef_pos[3:9] = left_6d
    eef_pos[9] = left_gripper  # left gripper position

    eef_pos[10:13] = right_eef_pose[0:3] / 1000  # right end-effector pose
    right_euler = right_eef_pose[3:6] / 180 * np.pi  # convert roll, pitch, yaw to radians
    right_6d = euler_to_rotate6d(right_euler, "xyz")
    eef_pos[13:19] = right_6d
    eef_pos[19] = right_gripper  # right gripper position

    return eef_pos


@dataclasses.dataclass(frozen=True)
class AgilexInputs(transforms.DataTransformFn):
    """Inputs for the Agilex policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    use_ee6d: bool = False

    def __call__(self, data: dict) -> dict:
        data = _decode_agilex(data)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        state = np.asarray(data["state"])
        if self.use_ee6d:
            if len(state.shape) == 1:
                state = joint_to_ee6d(state)
            elif len(state.shape) == 2:
                ee6d_state = np.zeros((state.shape[0], 20))
                for i in range(state.shape[0]):
                    ee6d_state[i] = joint_to_ee6d(state[i])
                state = ee6d_state
            else:
                raise ValueError(f"Expected state to have shape (14,) or (N, 14), got {state.shape}")

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if self.use_ee6d:
                assert len(actions.shape) == 2, f"Expected actions to have shape (N, 14), got {actions.shape}"
                ee6d_actions = np.zeros((actions.shape[0], 20))
                for i in range(actions.shape[0]):
                    ee6d_actions[i] = joint_to_ee6d(actions[i])
                actions = ee6d_actions
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "delay" in data:
            inputs["delay"] = data["delay"]

        if "action_prefix" in data:
            inputs["action_prefix"] = data["action_prefix"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AgilexOutputs(transforms.DataTransformFn):
    """Outputs for the Agilex policy."""

    use_ee6d: bool = False

    def __call__(self, data: dict) -> dict:
        if self.use_ee6d:
            actions = np.asarray(data["actions"][:, :20])
        else:
            # Only return the first 14 dims.
            actions = np.asarray(data["actions"][:, :14])
        return {"actions": actions}


def _decode_agilex(data: dict) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        if img.ndim == 3:
            img = img[None]
        return einops.rearrange(img, "t c h w -> t h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data
