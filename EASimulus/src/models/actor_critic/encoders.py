from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
from torch import nn as nn

from models.embedding import make_embeddings


class ObsEncoderBase(ABC, nn.Module):
    @abstractmethod
    def build_another(self):
        pass

    @property
    @abstractmethod
    def out_dim(self) -> int:
        pass


class ImageLatentObsEncoder(ObsEncoderBase):
    """
    Assumes obs latents are produced by VQVAE (ImageTokenizer).
    """
    def __init__(
            self, tokens_per_obs: int, embed_dim: int, num_layers: int, out_dim: Optional[int] = None, device=None,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tokens_per_obs = tokens_per_obs
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self._out_dim_orig = out_dim
        self.device = device

        self._flattened_dim = self.tokens_per_obs * self.embed_dim // (2 ** self.num_layers)
        self._out_dim = out_dim if out_dim is not None else self._flattened_dim
        self.model = self._build_cnn()

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def build_another(self):
        return ImageLatentObsEncoder(
            tokens_per_obs=self.tokens_per_obs,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            out_dim=self._out_dim_orig,
            device=self.device,
        )

    def _build_cnn(self):
        convs = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(
                    self.embed_dim // (2 ** i),
                    self.embed_dim // (2 ** (i + 1)),
                    3,
                    1,
                    1,
                    device=self.device
                ),
                nn.SiLU(),
            )
            for i in range(self.num_layers)
        ])

        convs.append(nn.Flatten(start_dim=1))

        if self._out_dim_orig is not None:
            in_features = self.tokens_per_obs * self.embed_dim // (2 ** self.num_layers)
            dense = nn.Sequential(
                nn.Linear(in_features, self.out_dim, device=self.device),
                nn.SiLU(),
            )

            convs.append(dense)
        return convs

    def forward(self, obs):
        assert obs.ndim == 4 and obs.shape[1] == self.embed_dim
        return self.model(obs)


class VectorObsEncoder(ObsEncoderBase):
    def __init__(self, obs_dim: int, out_dim: Optional[int] = None, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_dim = obs_dim
        self._out_dim_orig = out_dim
        self.device = device

        self._out_dim = obs_dim if out_dim is None else out_dim
        self.model = nn.Linear(obs_dim, out_dim, device=device) if out_dim is not None else nn.Identity()

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, obs):
        return self.model(obs)

    def build_another(self):
        return VectorObsEncoder(self.obs_dim, self._out_dim_orig, device=self.device)


class TokenObsEncoder(ObsEncoderBase):

    def __init__(self, nvec: np.ndarray, embed_dim: int, device=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nvec = nvec
        self.embed_dim = embed_dim
        self.device = device

        vocab_sizes = nvec[0]
        self.embeddings = make_embeddings(vocab_sizes=vocab_sizes, embed_dim=embed_dim, device=device)
        self._out_dim = nvec.shape[0] * embed_dim

    def forward(self, obs):
        return self.embeddings(obs).flatten(start_dim=1)

    def build_another(self):
        return TokenObsEncoder(nvec=self.nvec, embed_dim=self.embed_dim, device=self.device)

    @property
    def out_dim(self) -> int:
        return self._out_dim
