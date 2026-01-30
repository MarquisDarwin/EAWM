from typing import Optional, Tuple, Dict, Any, Union, List

import numpy as np
import gymnasium.spaces
from gymnasium.core import ObsType, WrapperObsType

from utils import ObsModality
from envs.wrappers.multi_modal import MultiModalObsWrapper, DictObsWrapper


def make_kinetix(id: str = "Kinetix-Pixels-MultiDiscrete-v1", is_train: bool = True, frame_size: int = 64, frame_skip: int = 2,tokenizer_config=None):
    from kinetix.environment.env import make_kinetix_env_from_name
    from kinetix.environment.env_state import StaticEnvParams, EnvParams
    from kinetix.environment.wrappers import AutoResetWrapper, UnderspecifiedToGymnaxWrapper, DenseRewardWrapper, AutoReplayWrapper
    from kinetix.util.config import generate_params_from_config
    from kinetix.environment.ued.ued import (
        make_reset_train_function_with_list_of_levels,
        make_reset_train_function_with_mutations,
    )
    # from gymnax.wrappers import GymnaxToGymWrapper
    from envs.wrappers.atari import ResizeObsWrapper

    class FixedAutoResetWrapper(AutoResetWrapper):
        def observation_space(self, params):
            return self._env.observation_space(params)

    class FixedAutoReplayWrapper(AutoReplayWrapper):
        def observation_space(self, params):
            return self._env.observation_space(params)

    class FixedUnderspecifiedToGymnaxWrapper(UnderspecifiedToGymnaxWrapper):
        def observation_space(self, params):
            return self._env.observation_space(params)

    import jax.random
    from gymnax.environments import environment
    from gymnax.environments import spaces
    import chex
    import gymnasium as gym
    from gymnasium import core
    from kinetix.environment.env import MultiDiscrete

    class GymnaxToGymWrapper(gym.Env[core.ObsType, core.ActType]):
        """Wrap Gymnax environment as OOP Gym environment."""

        def __init__(
                self,
                env: environment.Environment,
                params: Optional[environment.EnvParams] = None,
                seed: Optional[int] = None,
        ):
            """Wrap Gymnax environment as OOP Gym environment.


            Args:
                env: Gymnax Environment instance
                params: If provided, gymnax EnvParams for environment (otherwise uses
                  default)
                seed: If provided, seed for JAX PRNG (otherwise picks 0)
            """
            super().__init__()
            self._env = env
            self.env_params = params if params is not None else env.default_params
            self.metadata.update(
                {
                    "name": env.name,
                    "render_modes": (
                        ["human", "rgb_array"] if hasattr(env, "render") else []
                    ),
                }
            )
            self.rng: chex.PRNGKey = jax.random.PRNGKey(0)  # Placeholder
            self._seed(seed)
            _, self.env_state = self._env.reset(self.rng, self.env_params)

        @property
        def action_space(self):
            """Dynamically adjust action space depending on params."""
            if isinstance(self._env.action_space(self.env_params), MultiDiscrete):
                return gymnasium.spaces.MultiDiscrete(
                    nvec=self._env.action_space(self.env_params).number_of_dims_per_distribution)
            return spaces.gymnax_space_to_gym_space(self._env.action_space(self.env_params))

        @property
        def observation_space(self):
            """Dynamically adjust state space depending on params."""
            return spaces.gymnax_space_to_gym_space(
                self._env.observation_space(self.env_params)
            )

        def _seed(self, seed: Optional[int] = None):
            """Set RNG seed (or use 0)."""
            self.rng = jax.random.PRNGKey(seed or 0)

        def step(
                self, action: core.ActType
        ) -> Tuple[core.ObsType, float, bool, bool, Dict[Any, Any]]:
            """Step environment, follow new step API."""
            self.rng, step_key = jax.random.split(self.rng)
            o, self.env_state, r, d, info = self._env.step(
                step_key, self.env_state, action, self.env_params
            )
            return o, r, d, d, info

        def reset(
                self,
                *,
                seed: Optional[int] = None,
                return_info: bool = False,
                options: Optional[Any] = None,  # dict
        ) -> Tuple[core.ObsType, Any]:  # dict]:
            """Reset environment, update parameters and seed if provided."""
            if seed is not None:
                self._seed(seed)
            if options is not None:
                self.env_params = options.get(
                    "env_params", self.env_params
                )  # Allow changing environment parameters on reset
            self.rng, reset_key = jax.random.split(self.rng)
            o, self.env_state = self._env.reset(reset_key, self.env_params)
            return o, {}

        def render(
                self, mode="human"
        ) -> Optional[Union[core.RenderFrame, List[core.RenderFrame]]]:
            """use underlying environment rendering if it exists, otherwise return None."""
            return getattr(self._env, "render", lambda x, y: None)(
                self.env_state, self.env_params
            )

    cfg = {
        "num_polygons": 5,
        "num_circles": 2,
        "num_joints": 1,
        "num_thrusters": 1,
        "env_size_name": "s",
        "num_motor_bindings": 4,
        "num_thruster_bindings": 2,
        "env_size_type": "predefined",
        "frame_skip": frame_skip,
    }
    env_params, static_env_params = generate_params_from_config(cfg)

    # static_env_params = StaticEnvParams()
    static_env_params = static_env_params.replace(
        # screen_dim=(frame_size, frame_size),
        downscale=1,
        frame_skip=frame_skip,
    )

    # env_params = EnvParams()
    env_params = env_params.replace(
        dense_reward_scale=2.0,  # following the original config
    )

    env = make_kinetix_env_from_name(id, static_env_params=static_env_params)

    if is_train:
        reset_func = make_reset_train_function_with_mutations(
            env.physics_engine, env_params, env.static_env_params, config=cfg
        )
        env = FixedAutoResetWrapper(env, reset_func)
    else:
        # reset_func = make_reset_train_function_with_list_of_levels(
        #     cfg, cfg["train_levels_list"], static_env_params, is_loading_train_levels=True
        # )
        env = FixedAutoReplayWrapper(env)


    env = FixedUnderspecifiedToGymnaxWrapper(env)
    env = DenseRewardWrapper(env)
    env = GymnaxToGymWrapper(env)
    env = KinetixWrapper(env)
    env = ResizeObsWrapper(env, (frame_size, frame_size))

    env = DictObsWrapper(env)
    env = MultiModalObsWrapper(env, obs_key_to_modality={DictObsWrapper.key: ObsModality.image})

    return env


class KinetixWrapper(gymnasium.ObservationWrapper):

    def observation(self, observation: ObsType) -> WrapperObsType:
        return (np.array(observation.image) * 255).astype(np.uint8)


if __name__ == '__main__':
    env = make_kinetix()
    obs, info = env.reset()
    ep_len = 0
    for _ in range(100):
        step = env.step(env.action_space.sample())
        env.render()
        ep_len += 1
        if step[2]:
            print(f"Episode length: {ep_len}")
            step = env.reset()
            ep_len = 0


