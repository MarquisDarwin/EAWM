from abc import ABC, abstractmethod

import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from utils import ObsModality
from utils.math import sym_log


class ObsProcessor(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def __call__(self, obs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def to_torch(self, obs: np.ndarray, device=None) -> Tensor:
        pass


class ImageObsProcessor(ObsProcessor):
    def __call__(self, obs: Tensor) -> Tensor:
        return obs.float().div(255)

    def to_torch(self, obs: np.ndarray, device=None) -> Tensor:
        assert obs.ndim == 5 and obs.shape[1] == 1, f"got shape {obs.shape}"
        # Currently only supporting a single image. it shouldn't be very difficult to generalize to more.
        # TODO: implement support for obs with more than 1 image (batched with batch dim > 1)
        obs = obs.squeeze(1)
        obs_th = torch.tensor(obs, dtype=torch.uint8, device=device)
        if obs.shape[-1] == 3:
            # swap to channels first:
            obs_th = rearrange(obs_th, '... h w c -> ... c h w')
        else:
            assert obs.shape[-3] == 3, f"Got {obs.shape}"
        return obs_th.contiguous()


class TokenObsProcessor(ObsProcessor):
    def __call__(self, obs: Tensor) -> Tensor:
        return obs.long()

    def to_torch(self, obs: np.ndarray, device=None) -> Tensor:
        return torch.tensor(obs, dtype=torch.long, device=device)


class VectorObsProcessor(ObsProcessor):
    def __call__(self, obs: Tensor) -> Tensor:
        return sym_log(obs.float())

    def to_torch(self, obs: np.ndarray, device=None) -> Tensor:
        return torch.tensor(obs, dtype=torch.float32, device=device)


def get_obs_processor(modality: ObsModality, *args, **kwargs) -> ObsProcessor:
    modality_to_obs_processor = {
        ObsModality.image: ImageObsProcessor(*args, **kwargs),
        ObsModality.vector: VectorObsProcessor(*args, **kwargs),
        ObsModality.token: TokenObsProcessor(*args, **kwargs),
        ObsModality.token_2d: TokenObsProcessor(*args, **kwargs),
    }
    return modality_to_obs_processor[modality]


class EMAScaler:
    def __init__(self, decay: float = 0.001, quantile: float = 0.95):
        self.decay = decay
        self.quantile = quantile
        self._estimate_high = None
        self._estimate_low = None

    @property
    def estimate_high(self):
        return self._estimate_high

    @property
    def estimate_low(self):
        return self._estimate_low

    @property
    def scale(self):
        return self.estimate_high - self.estimate_low

    def update(self, values: Tensor):
        high = torch.quantile(values, self.quantile)
        low = torch.quantile(values, 1 - self.quantile)

        if self._estimate_high is None or self._estimate_low is None:
            self._estimate_high = high
            self._estimate_low = low
        else:
            self._estimate_high = self.decay * high + (1 - self.decay) * self._estimate_high
            self._estimate_low = self.decay * low + (1 - self.decay) * self._estimate_low


class BufferScaler:
    def __init__(
            self,
            quantile: float = 0.975,
            window_size_limit: int = 500
    ):
        self.quantile = quantile
        self.window_size_limit = window_size_limit
        self._buffer = None
        self._estimate_high = None
        self._estimate_low = None

    @property
    def estimate_high(self):
        return self._estimate_high

    @property
    def estimate_low(self):
        return self._estimate_low

    @property
    def scale(self):
        return self.estimate_high - self.estimate_low

    def update(self, values: Tensor):
        values = values.detach().clone().unsqueeze(0)
        if self._buffer is None:
            self._buffer = values
        else:
            self._buffer = torch.cat((self._buffer[-self.window_size_limit:], values), dim=0)

        high = torch.quantile(self._buffer, self.quantile)
        low = torch.quantile(self._buffer, 1 - self.quantile)

        self._estimate_high = high
        self._estimate_low = low
