from collections import OrderedDict
from typing import Any, Optional, Union, NamedTuple

import numpy as np
import dm_env
from dm_env import specs, StepType
import gymnasium
from gymnasium.core import ObsType, WrapperObsType
from loguru import logger
import os

import sys

sys.path.append("/home/pzh/doc/M3/src")
from utils.types import ObsModality
from utils.math import sym_log_np
from envs.wrappers.multi_modal import MultiModalObsWrapper, DictObsWrapper


def make_dm_control(id: str, hsize, wsize, image_output, tokenizer_config):
    logger.info(f"ENV: {id}")
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale

    domain_name, task_name = id.split(sep="-")
    env = suite.load(domain_name=domain_name, task_name=task_name)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, 2)
    env = action_scale.Wrapper(
        env, minimum=env.action_spec().minimum, maximum=env.action_spec().maximum
    )
    env = ExtendedTimeStepWrapper(env)
    env = DMCWrapper(env, domain_name, hsize, wsize, image_output, tokenizer_config)
    obs_key_to_modality = {}
    if tokenizer_config is not None:
        if hasattr(tokenizer_config, "vector"):
            obs_key_to_modality["vector"] = ObsModality.vector
    if image_output:
        obs_key_to_modality["image"] = ObsModality.image

    env = MultiModalObsWrapper(env, obs_key_to_modality=obs_key_to_modality)

    return env


def _extract_space(spec) -> gymnasium.spaces.Box:
    if isinstance(spec, OrderedDict):
        spaces = [_extract_space(v) for v in spec.values()]
        low = np.concatenate([s.low.flatten() for s in spaces])
        high = np.concatenate([s.high.flatten() for s in spaces])
        return gymnasium.spaces.Box(low, high)
    elif isinstance(spec, specs.BoundedArray):
        return gymnasium.spaces.Box(spec.minimum, spec.maximum, shape=spec.shape)
    elif isinstance(spec, specs.Array):
        return gymnasium.spaces.Box(-np.inf, np.inf, shape=spec.shape)
    else:
        assert False, f"Unexpected spec type '{type(spec)}'."


def _dmc_dict_obs_to_np_obs(dmc_obs: dict) -> np.ndarray:
    return np.concatenate([o_i.flatten() for o_i in dmc_obs.values()])


class DMCWrapper(gymnasium.Env):

    def __init__(self, dmc_env, domain, hsize, wsize, image_output, tokenizer_config):
        super().__init__()
        self.dmc_env = dmc_env
        #        self.observation_space = _extract_space(self.dmc_env.observation_spec())
        self.action_space = _extract_space(self.dmc_env.action_spec())
        self.tokenizer_config = tokenizer_config
        self._height, self._width = hsize, wsize
        self.image_output = image_output
        self._size = ()
        self._camera_id = dict(quadruped=2).get(domain, 0)

    @property
    def observation_space(self):
        spaces = {}
        if self.tokenizer_config is not None:
            if hasattr(self.tokenizer_config, "vector"):
                spaces["vector"] = _extract_space(self.dmc_env.observation_spec())
        if self.image_output:
            spaces["image"] = gymnasium.spaces.Box(
                0, 255, (self._height, self._width, 3), dtype=np.uint8
            )
        return gymnasium.spaces.Dict(spaces)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        dmc_step_data = self.dmc_env.step(action)
        obs = {}
        if self.tokenizer_config is not None:
            if hasattr(self.tokenizer_config, "vector"):
                obs["vector"] = _dmc_dict_obs_to_np_obs(dmc_step_data.observation)
        if self.image_output:
            obs["image"] = self.render()
        truncated = dmc_step_data.step_type.last()
        terminated = False
        return obs, dmc_step_data.reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
        dmc_step_data = self.dmc_env.reset()
        obs = {}
        if self.tokenizer_config is not None:
            if hasattr(self.tokenizer_config, "vector"):
                obs["vector"] = _dmc_dict_obs_to_np_obs(dmc_step_data.observation)
        if self.image_output:
            obs["image"] = self.render()
        info = {}
        return obs, info

    def render(self, mode="rgb_array", height=None, width=None, camera_id=0):
        # taken from https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self.dmc_env.physics.render(
            height=height, width=width, camera_id=camera_id
        )


class SymLogObsWrapper(gymnasium.ObservationWrapper):

    def observation(self, observation: ObsType) -> WrapperObsType:
        return sym_log_np(observation)


# The following wrappers were taken from https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/envs/dmcontrol.py
class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def test_make_dm_control():
    env = make_dm_control("humanoid-walk")
    logger.info(f"Obs space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    s = env.reset()
    a = env.action_space.sample()
    s, r, _, _, info = env.step(a)


if __name__ == "__main__":
    test_make_dm_control()
