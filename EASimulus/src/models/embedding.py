from typing import Union
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from models.tokenizer import HardCodedVectorTokenizer
from utils import UniformVQ


def make_embeddings(vocab_sizes: Union[list[int], np.ndarray, int], embed_dim: int, device=None) -> nn.Module:
    if isinstance(vocab_sizes, int) or np.isscalar(vocab_sizes):
        return nn.Embedding(vocab_sizes, embedding_dim=embed_dim, device=device)
    else:
        return Token2dEmbedding(vocab_sizes=vocab_sizes, embed_dim=embed_dim, device=device)


def make_mlp(
        layer_dims, activation: str = 'silu', layer_norm: bool = True, linear_out: bool = True, device=None
) -> nn.Module:
    activation_dict = {
        'silu': nn.SiLU,
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'prelu': nn.PReLU,
        'sigmoid': nn.Sigmoid,
    }

    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1], device=device))
        if i < len(layer_dims) - 1 or not linear_out:
            if layer_norm:
                layers.append(nn.LayerNorm(layer_dims[i+1], device=device))
            layers.append(activation_dict[activation]())

    return nn.Sequential(*layers)


class MultiDiscreteEmbedding(nn.Module):
    """
    For a sequence of independent categorical variables, with separate vocabularies.
    Assume the sequence is along the last dimension.
    """

    def __init__(
            self, vocab_sizes: Union[list[int], tuple[int, ...], np.ndarray], embed_dim: int,
            device=None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size_i, embedding_dim=embed_dim, device=device) for vocab_size_i in vocab_sizes
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        assert tokens.dim() >= 2, f"Got only {tokens.dim()} dimensions."
        assert tokens.shape[-1] == len(self.embeddings), f"Got {tokens.shape[-1]} embeddings."

        embeddings = [self.embeddings[i](tokens[..., i]) for i in range(tokens.shape[-1])]

        return torch.stack(embeddings, dim=-2)


class Token2dEmbedding(nn.Module):
    """
    Given a matrix of input tokens, all tokens of each row contribute to a shared embedding.
    The embedding of each row is computed as the sum of embeddings of all tokens in that row.
    Each column has its own embedding table.
    """

    def __init__(self, vocab_sizes: Union[list[int], np.ndarray], embed_dim: int, device=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embeddings = MultiDiscreteEmbedding(vocab_sizes=vocab_sizes, embed_dim=embed_dim, device=device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        assert tokens.dim() >= 3, f"Got only {tokens.dim()} dimensions."

        return self.embeddings(tokens).mean(dim=-2)


class ActionEncoder(nn.Module, ABC):

    @property
    @abstractmethod
    def action_sequence_length(self) -> int:
        pass

    @abstractmethod
    def embed_actions(self, actions: Tensor, **kwargs) -> Tensor:
        pass


class DiscreteActionEncoder(ActionEncoder):

    def __init__(self, num_actions: int, embed_dim: int, device=None, unknown_action=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(num_actions+1 if unknown_action else num_actions, embed_dim, device=device)

    @property
    def action_sequence_length(self) -> int:
        return 1

    def embed_actions(self, actions: Tensor, **kwargs) -> Tensor:
        assert actions.dim() == 2, f"got shape {actions.shape}"
        return self.embedding(rearrange(actions.long(), 'b l -> b l 1'))


class MultiDiscreteActionEncoder(ActionEncoder):
    def __init__(self, nvec: tuple[int, ...], embed_dim: int, device=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._nvec = nvec
        self.embedding = MultiDiscreteEmbedding(vocab_sizes=nvec, embed_dim=embed_dim, device=device)

    @property
    def action_sequence_length(self) -> int:
        return len(self._nvec)

    def embed_actions(self, actions: Tensor, **kwargs) -> Tensor:
        return self.embedding(actions)


class ContinuousActionEncoder(ActionEncoder):
    def __init__(
            self, action_dim: int, action_vocab_size: int, embed_dim: int, device=None,
            tokenize_actions: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenize_actions = tokenize_actions
        self._action_seq_len = action_dim if tokenize_actions else 1
        self.action_dim = action_dim
        self.action_projection = nn.Linear(action_dim, embed_dim, device=device)
        self.action_quantizer = UniformVQ(vmin=-1, vmax=1, support_size=action_vocab_size, device=device)
        self.embedding = nn.Embedding(action_vocab_size, embed_dim, device=device)

    @property
    def action_sequence_length(self) -> int:
        return self._action_seq_len

    def embed_actions(self, actions: Tensor, **kwargs) -> Tensor:
        assert actions.dim() == 3, f"Got shape {actions.shape}"
        if not self.tokenize_actions:
            return rearrange(self.action_projection(actions), 'b t e -> b t 1 e')
        else:
            action_tokens = self.action_quantizer.vector_to_tokens_pt(actions)  # b t d
            return self.embedding(action_tokens)  # b t d e
