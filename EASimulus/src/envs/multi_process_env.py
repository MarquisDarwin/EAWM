from dataclasses import astuple, dataclass
from enum import Enum
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import gymnasium.spaces
import numpy as np

from .done_tracker import DoneTrackerEnv


class MessageType(Enum):
    RESET = 0
    RESET_RETURN = 1
    STEP = 2
    STEP_RETURN = 3
    CLOSE = 4


@dataclass
class Message:
    type: MessageType
    content: Optional[Any] = None

    def __iter__(self) -> Iterator:
        return iter(astuple(self))


def child_env(child_id: int, env_fn: Callable, child_conn: Connection) -> None:
    np.random.seed(child_id + np.random.randint(0, 2 ** 31 - 1))
    env = env_fn()
    while True:
        message_type, content = child_conn.recv()
        if message_type == MessageType.RESET:
            obs, _ = env.reset()
            child_conn.send(Message(MessageType.RESET_RETURN, obs))
        elif message_type == MessageType.STEP:
            obs, rew, terminated, truncated, info = env.step(content)
            if terminated or truncated:
                obs, _ = env.reset()
            child_conn.send(Message(MessageType.STEP_RETURN, (obs, rew, terminated, truncated, info)))
        elif message_type == MessageType.CLOSE:
            child_conn.close()
            return
        else:
            raise NotImplementedError


def stack_obs(obs: list[np.ndarray]) -> np.ndarray:
    return np.stack(obs)


def stack_dict_obs(obs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: stack_obs([o[k] for o in obs]) for k in obs[0].keys()}


class MultiProcessEnv(DoneTrackerEnv):
    def __init__(self, env_fn: Callable, num_envs: int, should_wait_num_envs_ratio: float) -> None:
        super().__init__(num_envs)
        env = env_fn()
        self.modalities = env.modalities
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_actions = self.action_space.n if isinstance(self.action_space, gymnasium.spaces.Discrete) else None
        self.obs_stack_fn = stack_dict_obs if isinstance(env.observation_space, gymnasium.spaces.Dict) else stack_obs
        self.should_wait_num_envs_ratio = should_wait_num_envs_ratio
        self.processes, self.parent_conns = [], []
        for child_id in range(num_envs):
            parent_conn, child_conn = Pipe()
            self.parent_conns.append(parent_conn)
            p = Process(target=child_env, args=(child_id, env_fn, child_conn), daemon=True)
            self.processes.append(p)
        for p in self.processes:
            p.start()

    def should_reset(self) -> bool:
        return (self.num_envs_done / self.num_envs) >= self.should_wait_num_envs_ratio

    def _receive(self, check_type: Optional[MessageType] = None) -> List[Any]:
        messages = [parent_conn.recv() for parent_conn in self.parent_conns]
        if check_type is not None:
            assert all([m.type == check_type for m in messages])
        return [m.content for m in messages]

    def reset(self) -> tuple[Union[dict, np.ndarray], dict]:
        self.reset_done_tracker()
        for parent_conn in self.parent_conns:
            parent_conn.send(Message(MessageType.RESET))
        content = self._receive(check_type=MessageType.RESET_RETURN)
        return self.obs_stack_fn(content), {}

    def step(self, actions: np.ndarray) -> Tuple[Union[dict, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Any]:
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(Message(MessageType.STEP, action))
        content = self._receive(check_type=MessageType.STEP_RETURN)
        obs, rew, terminated, truncated, info = zip(*content)
        terminated = np.stack(terminated)
        truncated = np.stack(truncated)
        self.update_done_tracker(np.logical_or(terminated, truncated))
        return self.obs_stack_fn(obs), np.stack(rew), terminated, truncated, info

    def close(self) -> None:
        for parent_conn in self.parent_conns:
            parent_conn.send(Message(MessageType.CLOSE))
        for p in self.processes:
            p.join()
        for parent_conn in self.parent_conns:
            parent_conn.close()
