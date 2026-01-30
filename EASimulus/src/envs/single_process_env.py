from typing import Any, Tuple, Union

import gymnasium
import numpy as np

from .done_tracker import DoneTrackerEnv


def unsqueeze_obs(obs: np.ndarray) -> np.ndarray:
    return obs[None, ...]


def unsqueeze_dict_obs(obs: dict) -> dict:
    return {k: unsqueeze_obs(o) for k, o in obs.items()}


class SingleProcessEnv(DoneTrackerEnv):
    def __init__(self, env_fn):
        super().__init__(num_envs=1)
        self.env = env_fn()
        self.modalities = self.env.modalities
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.num_actions = self.action_space.n if isinstance(self.action_space, gymnasium.spaces.Discrete) else None
        self._obs_unsqueeze_fn = (
            unsqueeze_dict_obs if isinstance(self.env.observation_space, gymnasium.spaces.Dict) else unsqueeze_obs
        )

    def should_reset(self) -> bool:
        return self.num_envs_done == 1

    def reset(self) -> tuple[Union[dict, np.ndarray], dict]:
        self.reset_done_tracker()
        obs, info = self.env.reset()
        return self._obs_unsqueeze_fn(obs), info

    def step(self, action) -> Tuple[Union[dict, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Any]:
        obs, reward, terminated, truncated, info = self.env.step(action[0])  # action is supposed to be ndarray (1,)
        terminated = np.array([terminated])
        truncated = np.array([truncated])
        self.update_done_tracker(np.logical_or(terminated, truncated))
        return self._obs_unsqueeze_fn(obs), np.array([reward]), terminated, truncated, info

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()
