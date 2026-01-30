from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Optional

import torch
from torch import Tensor

from utils.types import MultiModalObs, ObsModality


@dataclass
class EpisodeMetrics:
    episode_length: int
    episode_return: float

    @property
    def rewards_per_step(self):
        return self.episode_return / self.episode_length


@dataclass
class Episode:
    observations: MultiModalObs
    actions: Tensor
    rewards: Tensor
    ends: Tensor
    mask_padding: Tensor
    last_info: dict[str, Any]
    events: MultiModalObs

    def __post_init__(self):
        obs_length_set = set([len(o_k) for o_k in self.observations.values()])
        assert len(obs_length_set) == 1
        assert obs_length_set.pop() == len(self.actions) == len(self.rewards) == len(self.ends) == len(self.mask_padding)
        assert set(self.observations.keys()).issubset(set(ObsModality))
        if self.ends.sum() > 0:
            idx_end = torch.argmax(self.ends) + 2  # include last obs
            if idx_end != len(self.actions):
                self.observations = {k: o_k[:idx_end] for k, o_k in self.observations.items()}
                self.actions = self.actions[:idx_end]
                self.rewards = self.rewards[:idx_end]
                self.ends = self.ends[:idx_end]
                self.mask_padding = self.mask_padding[:idx_end]
                if self.events is not None:
                    self.events = {k: m_k[:idx_end] for k, m_k in self.events.items()}

    def __len__(self) -> int:
        return self.actions.size(0)

    def merge(self, other: Episode) -> Episode:
        return Episode(
            {k: torch.cat((o_k, other.observations[k]), dim=0) for k, o_k in self.observations.items()},
            torch.cat((self.actions, other.actions), dim=0),
            torch.cat((self.rewards, other.rewards), dim=0),
            torch.cat((self.ends, other.ends), dim=0),
            torch.cat((self.mask_padding, other.mask_padding), dim=0),
            other.last_info,
            None if self.events is None else {k: torch.cat((m_k, other.events[k]), dim=0) for k, m_k in self.events.items()}
        )

    def segment(self, start: int, stop: int, should_pad: bool = False) -> Episode:
        assert start < len(self) and stop > 0 and start < stop, f"Got start={start}, stop={stop}, len(self)={len(self)}"
        padding_length_right = max(0, stop - len(self))
        padding_length_left = max(0, -start)
        assert padding_length_right == padding_length_left == 0 or should_pad

        def pad(x):
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        start = max(0, start)
        stop = min(len(self), stop)
        segment = Episode(
            {k: o_k[start:stop] for k, o_k in self.observations.items()},
            self.actions[start:stop],
            self.rewards[start:stop],
            self.ends[start:stop],
            self.mask_padding[start:stop],
            self.last_info,
            None if self.events is None else {k: m_k[start:stop] for k, m_k in self.events.items()}
        )

        segment.observations = {k: pad(o_k) for k, o_k in segment.observations.items()}
        segment.actions = pad(segment.actions)
        segment.rewards = pad(segment.rewards)
        segment.ends = pad(segment.ends)
        if self.events is not None:
            segment.events= {k: pad(m_k) for k, m_k in segment.events.items()}
        segment.mask_padding = torch.cat((torch.zeros(padding_length_left, dtype=torch.bool), segment.mask_padding, torch.zeros(padding_length_right, dtype=torch.bool)), dim=0)

        return segment

    def compute_metrics(self) -> EpisodeMetrics:
        return EpisodeMetrics(len(self), self.rewards.sum().item())

    def to_dict(self):
        obj = self.__dict__.copy()
        obj['observations'] = {k.name: v for k, v in obj['observations'].items()}
        return obj

    def save(self, path: Path) -> None:
        torch.save(self.to_dict(), path)

    @classmethod
    def from_dict(cls, dict_episode):
        dict_episode['observations'] = {ObsModality[k]: v for k, v in dict_episode['observations'].items()}
        dict_episode['ends'] = dict_episode['ends'].long()
        return Episode(**dict_episode)
