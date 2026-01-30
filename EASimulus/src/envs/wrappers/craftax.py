import os
from typing import SupportsFloat, Any

import numpy as np
import torch
import gymnasium
from gymnasium import Env
from gymnasium.core import ObsType, ActType, WrapperObsType, WrapperActType

from envs.wrappers.multi_modal import MultiModalObsWrapper, DictObsWrapper
from utils import ObsModality


os.environ['JAX_PLATFORMS'] = 'cpu'


def make_craftax(id: str = "Craftax-Symbolic-v1",tokenizer_config=None):
    from craftax.craftax_env import make_craftax_env_from_name
    from gymnax.wrappers import GymnaxToGymWrapper

    env = make_craftax_env_from_name(id, auto_reset=True)
    env = GymnaxToGymWrapper(env)
    env = CraftaxWrapper(env)
    env = InfoWrapper(env)
    env = MultiModalObsWrapper(env, obs_key_to_modality={
        'map': ObsModality.token_2d,
        'stats': ObsModality.vector,
        'direction': ObsModality.token,
    })

    return env


class CraftaxWrapper(gymnasium.ObservationWrapper):

    def __init__(self, env: Env[ObsType, ActType], light_level_step_size: float = 0.05):
        super().__init__(env)
        self.inventory_numel = 51
        self._direction_index = 31
        self._per_cell_sizes = [37, 5, 8*5, 1]
        self._per_cell_vocab_sizes = self._per_cell_sizes[:-1] + [1//light_level_step_size + 1]
        self._map_num_cells = 9 * 11
        self._light_level_step_size = light_level_step_size

        self.observation_space = gymnasium.spaces.Dict({
            'map': gymnasium.spaces.MultiDiscrete(
                np.broadcast_to(np.array(self._per_cell_vocab_sizes).reshape((1, 4)), (self._map_num_cells, 4))
            ),
            'stats': gymnasium.spaces.Box(low=0, high=2.1, shape=(47,)),
            'direction': gymnasium.spaces.MultiDiscrete(nvec=(4,))
        })

    def observation(self, observation: ObsType) -> dict[str, np.ndarray]:
        observation = np.array(observation)

        map_numel = sum(self._per_cell_sizes) * self._map_num_cells
        map_elements, inventory = np.split(observation, [map_numel])

        inventory1, direction, inventory2 = np.split(inventory, [self._direction_index, self._direction_index + 4])
        inventory = np.concatenate([inventory1, inventory2])
        direction_token = np.argmax(direction, keepdims=True)

        map_elements = map_elements.reshape((self._map_num_cells, sum(self._per_cell_sizes)))
        split_indices = [np.sum(self._per_cell_sizes[:i + 1]) for i in range(len(self._per_cell_sizes)-1)]
        block_type, item_type, mob_type, light_level = np.split(map_elements, split_indices, axis=1)

        block_token = np.argmax(block_type, axis=1).astype(np.int32)
        item_token = np.argmax(item_type, axis=1).astype(np.int32)
        mob_token = np.argmax(mob_type, axis=1).astype(np.int32)
        light_token = np.squeeze(light_level, axis=1) // self._light_level_step_size

        map_tokens = np.stack([block_token, item_token, mob_token, light_token.astype(np.int32)], axis=1, dtype=np.int32)

        obs = {
            'map': map_tokens,
            'stats': inventory,
            'direction': direction_token,
        }

        return obs


class InfoWrapper(gymnasium.Wrapper):

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        info = {k: np.array(v).item() for k, v in info.items()}
        return obs, reward, terminated, truncated, info
