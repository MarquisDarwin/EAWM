import os

import gymnasium
from gymnasium import Env
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Box
import numpy as np
from loguru import logger
import tqdm

from envs.wrappers.multi_modal import MultiModalObsWrapper


def make_bigym(id: str, resolution: int = 64, control_frequency: int = 50, max_episode_steps: int = 400, headless: bool = True,tokenizer_config=None):
    if headless:
        os.environ['MUJOCO_GL'] = "osmesa"

    if id == 'flip_cup':
        env = make_flip_cup(resolution=resolution)
    elif id == 'reach_target_dual':
        env = get_reach_target_dual_env(resolution=resolution)
    elif id == 'wall_cupboard_close':
        env = get_wall_cupboard_close_env(resolution=resolution)
    else:
        raise ValueError(f'Unsupported environment "{id}"')

    env = MultiModalObsWrapper(env)
    env = NormalizedActionsWrapper(env)
    env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def make_flip_cup(resolution: int = 64):
    from bigym.action_modes import JointPositionActionMode, PelvisDof, TorqueActionMode
    from bigym.utils.observation_config import ObservationConfig, CameraConfig
    from bigym.envs.manipulation import FlipCup

    control_frequency = 50

    return FlipCup(
        action_mode=JointPositionActionMode(absolute=True, floating_base=True,
                                            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]),
        # TorqueActionMode(floating_base=True),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(
                    name="head",
                    rgb=True,
                    depth=False,
                    resolution=(resolution, resolution),
                )
            ],
        ),
        render_mode='rgb_array',
        control_frequency=control_frequency
    )


def get_reach_target_dual_env(resolution: int = 64):
    from bigym.action_modes import JointPositionActionMode, PelvisDof, TorqueActionMode
    from bigym.utils.observation_config import ObservationConfig, CameraConfig
    from bigym.envs.reach_target import ReachTargetDual

    control_frequency = 50
    env = ReachTargetDual(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True,
                                            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.RZ]),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", resolution=(resolution, resolution)),
                # CameraConfig("left_wrist", resolution=(resolution, resolution)),
                # CameraConfig("right_wrist", resolution=(resolution, resolution)),
            ]
        ),
        render_mode="rgb_array",
    )
    return env


def get_wall_cupboard_close_env(resolution: int = 64):
    from bigym.action_modes import JointPositionActionMode, PelvisDof, TorqueActionMode
    from bigym.utils.observation_config import ObservationConfig, CameraConfig
    from bigym.envs.cupboards import WallCupboardClose

    control_frequency = 50
    env = WallCupboardClose(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True,
                                            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", resolution=(resolution, resolution)),
                # CameraConfig("left_wrist", resolution=(resolution, resolution)),
                # CameraConfig("right_wrist", resolution=(resolution, resolution)),
            ]
        ),
        render_mode="rgb_array",
    )
    return env


class NormalizedActionsWrapper(gymnasium.ActionWrapper):

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)
        self._orig_action_space = env.action_space
        assert isinstance(self._orig_action_space, Box), f"Got {self._orig_action_space}"
        assert self._orig_action_space.is_bounded()
        self.action_space = Box(low=-1, high=1, shape=env.action_space.shape, dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        orig_low, orig_high = self._orig_action_space.low, self._orig_action_space.high
        orig_diff = orig_high - orig_low
        diff = self.action_space.high - self.action_space.low
        raw_action = orig_low + (action - self.action_space.low) * orig_diff / diff
        raw_action = np.clip(raw_action, orig_low, orig_high)
        return raw_action


def sanity_check():
    env = make_bigym('flip_cup')
    logger.debug(f"obs space: {env.observation_space}")
    logger.debug(f"action space: {env.action_space}")
    obs, info = env.reset()
    logger.debug(f"obs:{obs}")

    pbar = tqdm.trange(100)
    for i in pbar:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # env.render()
        # pbar.set_description(f"obs:{obs}; reward{reward}; terminated{terminated}; truncated{truncated}; info:{info}")


if __name__ == '__main__':
    sanity_check()



