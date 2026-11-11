"""Zu7 data transforms for the pinned OpenPI pi0.5 implementation."""

import dataclasses

import numpy as np
from openpi import transforms
from openpi.training import config as openpi_config


JOINT_DELTA_MASK = (True,) * 6 + (False,) + (True,) * 6 + (False,)
CAMERA_KEYS = {
    "base_0_rgb": "image",
    "left_wrist_0_rgb": "left_wrist_image",
    "right_wrist_0_rgb": "right_wrist_image",
}


def _image(value):
    image = np.asarray(value)
    if image.shape == (3, 240, 320):
        image = np.moveaxis(image, 0, -1)
    if image.shape != (240, 320, 3):
        raise ValueError(f"Unexpected Zu7 image shape: {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Zu7 image values must be finite and in [0, 1]")
        image = (image * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError(f"Unexpected Zu7 image dtype: {image.dtype}")
    return image


@dataclasses.dataclass(frozen=True)
class Zu7Inputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # OpenPI DeltaActions/AbsoluteActions modify their arrays in place.
        # Own these buffers so cached dataset samples and caller observations stay intact.
        state = np.array(data["state"], dtype=np.float32, copy=True)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 14D measured Zu7 state")
        result = {
            "state": state,
            "image": {target: _image(data[source]) for target, source in CAMERA_KEYS.items()},
            "image_mask": {target: np.True_ for target in CAMERA_KEYS},
        }
        if "actions" in data:
            actions = np.array(data["actions"], dtype=np.float32, copy=True)
            if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
                raise ValueError("Expected finite [horizon, 14] Zu7 actions")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class Zu7Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < 14 or not np.isfinite(actions).all():
            raise ValueError("Expected finite OpenPI [horizon, >=14] action output")
        return {"actions": np.array(actions[:, :14], dtype=np.float32, copy=True)}


@dataclasses.dataclass(frozen=True)
class LeRobotZu7DataConfig(openpi_config.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        data_transforms = transforms.Group(
            inputs=[Zu7Inputs(), transforms.DeltaActions(JOINT_DELTA_MASK)],
            outputs=[transforms.AbsoluteActions(JOINT_DELTA_MASK), Zu7Outputs()],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=openpi_config.ModelTransformFactory()(model_config),
            action_sequence_keys=("actions",),
        )
