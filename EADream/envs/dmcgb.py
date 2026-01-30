import gym
import numpy as np
import augmentations
import torch
class DeepMindControl:
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0, max_episode_steps=500,is_aug=False):
        domain, task = name.split("_", 1)
        self.is_aug=is_aug
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if domain == "ball":  # Only domain with multiple words.
            domain = "ball_in_cup"
            task="catch"
        if isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(
                domain,
                task,
                task_kwargs={"random": seed},
            )
            #print(vars(self._env))
        else:
            assert task is None
            self._env = domain()
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self._domain_name = domain
        self._task_name = task
        self._max_episode_steps=max_episode_steps
        self.reward_range = [-np.inf, np.inf]
    def __getattr__(self, name):
        return getattr(self._env, name)
    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        if self.is_aug:
            obs["image"] = self.apply_strong_aug(torch.Tensor(obs["image"].copy()).unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(0).detach().numpy()
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        self.apply_strong_aug = augmentations.compose_augs()
        obs["image"] = self.render()
        if self.is_aug:
            obs["image"] = self.apply_strong_aug(torch.Tensor(obs["image"].copy()).unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(0).detach().numpy()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
