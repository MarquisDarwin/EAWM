"""
Credits to https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""
from typing import Tuple

# import gym
import numpy as np
from PIL import Image
import gymnasium
from loguru import logger

from envs.wrappers.multi_modal import MultiModalObsWrapper, DictObsWrapper
from utils.types import ObsModality


def make_atari(id, hsize=64,wsize=64, max_episode_steps=None, noop_max=30, frame_skip=4, done_on_life_loss=False, clip_reward=False,tokenizer_config=None):
    logger.info(f"ENV: {id}")
    import ale_py
    env = gymnasium.make(id)
    # assert 'NoFrameskip' in env.spec.id or 'Frameskip' not in env.spec
    env = ResizeObsWrapper(env, (hsize,wsize))
    if clip_reward:
        env = RewardClippingWrapper(env)
    if max_episode_steps is not None:
        env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if noop_max is not None:
        env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    if done_on_life_loss:
        env = EpisodicLifeEnv(env)

    env = DictObsWrapper(env)
    env = MultiModalObsWrapper(env, obs_key_to_modality={DictObsWrapper.key: ObsModality.image})
    return env


# class Gym2GymnasiumWrapper(gym.Wrapper):
#     def __init__(self, env: gym.Env):
#         super().__init__(env)
#
#     def step(self, action):
#         assert isinstance(self.env, gym.Env)
#         obs, reward, terminated, info = self.env.step(action)
#         return obs, reward, terminated, False, info
#
#     def reset(self, **kwargs):
#         return self.env.reset(), {}
#
#
# class Gymnasium2GymWrapper(gym.Wrapper):
#
#     def __init__(self, env: gymnasium.Env):
#         super().__init__(env)
#
#     def step(self, action):
#         assert isinstance(self.env, gymnasium.Env)
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         done = terminated or truncated
#         return obs, reward, done, info
#
#     def reset(self):
#         return self.env.reset()[0]


class ResizeObsWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env: gymnasium.Env, size: Tuple[int, int]) -> None:
        gymnasium.ObservationWrapper.__init__(self, env)
        self.size = tuple(size)
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)
        self.unwrapped.original_obs = None

    def resize(self, obs: np.ndarray):
        img = Image.fromarray(obs)
        img = img.resize((self.size[1],self.size[0]), Image.BILINEAR)
        return np.array(img)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.unwrapped.original_obs = observation
        return self.resize(observation)


class RewardClippingWrapper(gymnasium.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)


class NoopResetEnv(gymnasium.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gymnasium.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        obs, info = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0

        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gymnasium.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gymnasium.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises terminated.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gymnasium.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gymnasium.Wrapper.__init__(self, env)
        assert skip > 0
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        assert self._skip > 0
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                self._obs_buffer[0] = obs
                self._obs_buffer[1] = obs
                break

        self.max_frame = self._obs_buffer.max(axis=0)
        # self.max_frame = 0.25 * self._obs_buffer[0] + 0.75 * self._obs_buffer[1]

        return self.max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


if __name__ == '__main__':
    from envs.single_process_env import SingleProcessEnv
    raw_env = make_atari("DemonAttackNoFrameskip-v4", max_episode_steps=20000, noop_max=30, frame_skip=4, done_on_life_loss=True, clip_reward=False)
    env_fn = lambda : raw_env
    env = SingleProcessEnv(env_fn)
    obs, info = env.reset()
    ep_len = 0
    for i in range(100000):
        #env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step([5])
        ep_len += 1
        if truncated[0] or terminated[0]:
            logger.info(f"terminated={terminated} ({[terminated[0]]}), truncated={truncated}; ep_len={ep_len}")
            assert env.should_reset()
            env.reset()
            assert not env.should_reset()
            ep_len = 0
        else:
            assert not env.should_reset()
