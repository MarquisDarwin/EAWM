from typing import Optional

import gymnasium
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium.core import ObsType, ActType, WrapperObsType

from utils import ObsModality


def auto_detect_modality(obs_space: gymnasium.spaces.Space) -> ObsModality:
    if np.issubdtype(obs_space.dtype, np.uint8) and len(obs_space.shape) in [2, 3]:
        return ObsModality.image
    elif np.issubdtype(obs_space.dtype, np.floating) and len(obs_space.shape) == 1:
        return ObsModality.vector
    elif np.issubdtype(obs_space.dtype, np.integer) and len(obs_space.shape) == 1:
        return ObsModality.token
    elif np.issubdtype(obs_space.dtype, np.integer) and len(obs_space.shape) == 2:
        return ObsModality.token_2d
    else:
        raise ValueError(f"Observation space '{obs_space}' is not supported or could not be detected automatically.")


class MultiModalObsWrapper(gymnasium.ObservationWrapper):
    """
    A general wrapper for handling environments with multiple modalities.
    Single modality environments are a special case of multimodal environments.
    """

    def __init__(self, env: Env[ObsType, ActType], obs_key_to_modality: Optional[dict[str, ObsModality]] = None):
        assert isinstance(env.observation_space, gymnasium.spaces.Dict)
        super().__init__(env)
        if obs_key_to_modality is None:
            obs_key_to_modality = {k: auto_detect_modality(v) for k, v in env.observation_space.items()}
        self.obs_key_to_modality = obs_key_to_modality
        self._modalities = set(obs_key_to_modality.values())
        self.observation_space = self._make_obs_space(env.observation_space)

    def _make_obs_space(self, orig_obs_space: gymnasium.spaces.Dict) -> gymnasium.spaces.Dict:
        obs_spaces = {modality_type: [] for modality_type in self.modalities}
        for k, v in orig_obs_space.spaces.items():
            obs_spaces[self.obs_key_to_modality[k]].append(v)

        new_obs_space = {}

        if ObsModality.image in obs_spaces:
            assert len(set([s.shape for s in obs_spaces[ObsModality.image]])) == 1, f"Currently, only uniform image size is supported."
            stacked_shape = (len(obs_spaces[ObsModality.image]), *obs_spaces[ObsModality.image][0].shape)
            new_obs_space[ObsModality.image] = Box(low=0, high=255, shape=stacked_shape, dtype=np.uint8)

        if ObsModality.vector in obs_spaces:
            lows = np.concatenate([s.low for s in obs_spaces[ObsModality.vector]])
            highs = np.concatenate([s.high for s in obs_spaces[ObsModality.vector]])
            new_obs_space[ObsModality.vector] = Box(low=lows, high=highs, dtype=np.float32)

        if ObsModality.token in obs_spaces:
            assert all([isinstance(s, (MultiDiscrete, Box)) for s in obs_spaces[ObsModality.token]])
            nvec = np.concatenate([s.nvec if isinstance(s, MultiDiscrete) else s.high for s in obs_spaces[ObsModality.token]])
            new_obs_space[ObsModality.token] = MultiDiscrete(nvec=nvec, dtype=np.int32)

        if ObsModality.token_2d in obs_spaces:
            # TODO: Support multiple token_2d
            assert len(obs_spaces[ObsModality.token_2d]) == 1, f"Currently, only a single token_2d input is supported."
            assert all([isinstance(s, MultiDiscrete) and s.nvec.ndim == 2 for s in obs_spaces[ObsModality.token_2d]])
            new_obs_space[ObsModality.token_2d] = obs_spaces[ObsModality.token_2d][0]

        return gymnasium.spaces.Dict(new_obs_space)

    @property
    def modalities(self) -> set[ObsModality]:
        return self._modalities

    def observation(self, observation: ObsType) -> dict[str, np.ndarray]:
        per_modality_obs = {modality_type: [] for modality_type in self.modalities}
        for k, v in observation.items():
            per_modality_obs[self.obs_key_to_modality[k]].append(v)

        obs = {}

        if ObsModality.image in per_modality_obs:
            obs[ObsModality.image] = np.stack(per_modality_obs[ObsModality.image])

        if ObsModality.vector in per_modality_obs:
            obs[ObsModality.vector] = np.concatenate(per_modality_obs[ObsModality.vector])

        if ObsModality.token in per_modality_obs:
            obs[ObsModality.token] = np.concatenate(per_modality_obs[ObsModality.token])

        if ObsModality.token_2d in per_modality_obs:
            assert len(per_modality_obs[ObsModality.token_2d]) == 1, f"Currently, only a single token_2d input is supported."
            obs[ObsModality.token_2d] = per_modality_obs[ObsModality.token_2d][0]

        return {k.name: v for k, v in obs.items()}


class DictObsWrapper(gymnasium.ObservationWrapper):
    """
    A dummy wrapper that makes a dict obs space so that the env is compatible with
    the multi-modality wrapper.
    """
    key = 'features'

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Dict({self.key: env.observation_space})

    def observation(self, observation: ObsType) -> WrapperObsType:
        return {self.key: observation}





