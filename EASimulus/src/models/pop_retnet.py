# Based on https://github.com/fkodom/yet-another-retnet/blob/main/yet_another_retnet/retention.py
from functools import lru_cache
from math import ceil, log
from typing import Union, Callable, Optional, List, Sequence, Tuple
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
import yet_another_retnet as yar
import yet_another_retnet.retention
from einops import rearrange, einsum
from torch import Tensor
from yet_another_retnet.retention import (
    ActivationString
)
from yet_another_retnet.retnet import RetNetDecoder


DECAY_SCALE_MIN_NUM_BLOCKS = 4
DECAY_SCALE_MAX_NUM_BLOCKS = 512


@lru_cache(maxsize=1)
def _build_decay_gammas(
        num_heads: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
) -> Tensor:
    """Decay values are different for each retention head, following the prescribed
    method in the paper.  Conceptually, I think of each head having a different
    "retention window", which is the effective number of steps back in time that
    the head can attend to.  Retention windows are effectively determined by
    these decay coefficients.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Section 3.1 (Setup)
    """
    if xmin is None:
        xmin = log(1 / 32)
    if xmax is None:
        xmax = log(1 / 512)
    x = torch.linspace(xmin, xmax, steps=num_heads, device=device, dtype=dtype)
    return 1 - x.exp_()


def _build_causal_mask(
        sequence_length: int,
        tokens_per_block: int,
        device: Optional[Union[torch.device, str]] = None
) -> Tensor:
    block_mask = torch.ones(tokens_per_block, tokens_per_block, dtype=torch.bool, device=device)
    block_mask[:-1, -1] = 0  # obs tokens cannot attend the last element (action)

    assert sequence_length % tokens_per_block == 0 or sequence_length <= tokens_per_block
    if sequence_length <= tokens_per_block:
        return block_mask[:sequence_length, :sequence_length]
    else:
        num_blocks = int(ceil(sequence_length / tokens_per_block))
        tril = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.bool))
        mask = torch.kron(tril, block_mask)
        return mask[:sequence_length, :sequence_length]


@lru_cache(maxsize=1)
def _build_causal_decay_mask(
        num_heads: int,
        query_length: int,
        key_length: int,
        tokens_per_block: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        decay_scale_min_num_blocks: int = DECAY_SCALE_MIN_NUM_BLOCKS,
        decay_scale_max_num_blocks: int = DECAY_SCALE_MAX_NUM_BLOCKS,
) -> Tensor:
    """The decay mask is one of the key components that makes *parallel* retention
    equivalent to *recurrent* retention.  The decay coefficients are pre-computed
    and applied to the similarity matrix at once, rather than being applied to
    each element in the recurrent formulation.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Equation 5
    """
    decay_gammas = _build_decay_gammas(num_heads=num_heads, device=device, dtype=dtype,
                                       xmin=log(1 / (decay_scale_min_num_blocks * tokens_per_block)),
                                       xmax=log(1 / (decay_scale_max_num_blocks * tokens_per_block)))

    query_pos = torch.arange(query_length, device=device, dtype=dtype).unsqueeze_(-1)
    key_pos = torch.arange(key_length, device=device, dtype=dtype).unsqueeze_(0)
    distance = torch.abs(query_pos - key_pos)

    distance = rearrange(distance, "n s -> () n s")
    decay_gammas = rearrange(decay_gammas, "h -> h () ()")
    # NOTE: Keep only the lower-triangular elements (including the diagonal), so that
    # *future* keys cannot affect the current query. The .tril() method is not yet
    # implemented for bfloat16 dtypes, so we use .masked_fill_() instead,
    # which is slightly slower.
    # Thanks to @Doraemonzzz for catching this bug!
    decay_mask = decay_gammas ** distance

    # build the causal mask (tokens within an observations can attend to each other, not necessarily AR):
    assert key_length == query_length, f"Got different key, query lengths: {key_length}, {query_length}."
    causal_mask = _build_causal_mask(
        sequence_length=key_length,
        tokens_per_block=tokens_per_block,
        device=device
    )
    return decay_mask.masked_fill_(torch.logical_not(causal_mask), 0)


@lru_cache(maxsize=1)
def _build_autoregressive_decay_mask(
        num_heads: int,
        query_length: int,
        key_length: int,
        tokens_per_block: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        decay_scale_min_num_blocks: int = DECAY_SCALE_MIN_NUM_BLOCKS,
        decay_scale_max_num_blocks: int = DECAY_SCALE_MAX_NUM_BLOCKS,
) -> Tensor:
    """The decay mask is one of the key components that makes *parallel* retention
    equivalent to *recurrent* retention.  The decay coefficients are pre-computed
    and applied to the similarity matrix at once, rather than being applied to
    each element in the recurrent formulation.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Equation 5
    """
    decay_gammas = _build_decay_gammas(num_heads=num_heads, device=device, dtype=dtype,
                                       xmin=log(1 / (decay_scale_min_num_blocks * tokens_per_block)),
                                       xmax=log(1 / (decay_scale_max_num_blocks * tokens_per_block)))

    query_pos = torch.arange(query_length, device=device, dtype=dtype).unsqueeze_(-1)
    key_pos = torch.arange(key_length, device=device, dtype=dtype).unsqueeze_(0)
    distance = torch.abs(query_pos - key_pos)

    distance = rearrange(distance, "n s -> () n s")
    decay_gammas = rearrange(decay_gammas, "h -> h () ()")
    # NOTE: Keep only the lower-triangular elements (including the diagonal), so that
    # *future* keys cannot affect the current query. The .tril() method is not yet
    # implemented for bfloat16 dtypes, so we use .masked_fill_() instead,
    # which is slightly slower.
    # Thanks to @Doraemonzzz for catching this bug!
    decay_mask = decay_gammas ** distance
    future_mask = torch.ones_like(decay_mask, dtype=torch.bool).triu_(diagonal=1)
    return decay_mask.masked_fill_(future_mask, 0)


def _build_decay_mask(
        num_heads: int,
        query_length: int,
        key_length: int,
        tokens_per_block: int,
        mask_type: str = 'autoregressive',
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        decay_scale_min_num_blocks: int = DECAY_SCALE_MIN_NUM_BLOCKS,
        decay_scale_max_num_blocks: int = DECAY_SCALE_MAX_NUM_BLOCKS,
):
    if mask_type == 'autoregressive':
        return _build_autoregressive_decay_mask(
            num_heads=num_heads,
            query_length=query_length,
            key_length=key_length,
            tokens_per_block=tokens_per_block,
            device=device,
            dtype=dtype,
            decay_scale_min_num_blocks=decay_scale_min_num_blocks,
            decay_scale_max_num_blocks=decay_scale_max_num_blocks
        )
    elif mask_type == 'causal':
        return _build_causal_decay_mask(
            num_heads=num_heads,
            query_length=query_length,
            key_length=key_length,
            tokens_per_block=tokens_per_block,
            device=device,
            dtype=dtype,
            decay_scale_min_num_blocks=decay_scale_min_num_blocks,
            decay_scale_max_num_blocks=decay_scale_max_num_blocks
        )
    else:
        assert False, f"Mask type '{mask_type}' is not supported."


def trim_pred_tokens_suffix(x: Tensor, tokens_per_block: int):
    assert x.dim() == 4, f"Got {x.shape}"

    seq_len = x.shape[2]
    residue = seq_len % tokens_per_block
    return x[:, :, :-residue] if residue > 0 else x


def retention_chunkwise_per_block_states(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        prev_state: Optional[Tensor],
        tokens_per_block: int,
        mask_type: str = 'autoregressive',
        scale: Optional[float] = None,
        decay_scale_min_num_blocks: int = DECAY_SCALE_MIN_NUM_BLOCKS,
        decay_scale_max_num_blocks: int = DECAY_SCALE_MAX_NUM_BLOCKS,
) -> Tuple[Tensor, Tensor]:
    decay_gammas = _build_decay_gammas(
        num_heads=query.shape[1], device=query.device, dtype=query.dtype,
        xmin=log(1 / (decay_scale_min_num_blocks * tokens_per_block)),
        xmax=log(1 / (decay_scale_max_num_blocks * tokens_per_block)),
    )
    decay_mask = _build_decay_mask(
        num_heads=query.shape[1],
        query_length=query.shape[2],
        key_length=key.shape[2],
        tokens_per_block=tokens_per_block,
        mask_type=mask_type,
        device=query.device,
        dtype=query.dtype,
        decay_scale_min_num_blocks=decay_scale_min_num_blocks,
        decay_scale_max_num_blocks=decay_scale_max_num_blocks
    )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - n / s: seq_length
    # - d: head_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    # intra-chunk (same as parallel retention)
    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
    retention = einsum(similarity, value, "b h n s, b h s d -> b h n d")

    # cross-chunk (derived from recurrent retention)
    decay_gammas = rearrange(decay_gammas, "h -> () h () ()")
    inner_pos = rearrange(
        torch.arange(key.size(2), device=key.device, dtype=key.dtype) + 1,
        "n -> () () n ()",
    )
    per_frame_inner_pos = rearrange(
        torch.arange(tokens_per_block, device=key.device, dtype=key.dtype) + 1,
        "k1 -> () () k1 ()",
    )

    per_frame_state_decays = decay_gammas ** (tokens_per_block - per_frame_inner_pos)

    # For inference, where we append a pred tokens suffix (shouldn't be included in the state update):
    # key = trim_pred_tokens_suffix(key, tokens_per_block)
    # value = trim_pred_tokens_suffix(value, tokens_per_block)

    key = rearrange(key, 'b h (t k1) d -> b h t k1 d', k1=tokens_per_block)
    value = rearrange(value, 'b h (t k1) d -> b h t k1 d', k1=tokens_per_block)
    discounted_key = einsum(key, per_frame_state_decays, 'b h t k1 d, _ h k1 _ -> b h t k1 d')
    state = einsum(discounted_key, value, 'b h t k1 d1, b h t k1 d2 -> b h t d1 d2')

    per_frame_decay_gammas = decay_gammas ** tokens_per_block
    if prev_state is not None:
        state[:, :, 0] = state[:, :, 0] + per_frame_decay_gammas * prev_state
    for i in range(1, state.shape[2]):
        state[:, :, i] = state[:, :, i] + per_frame_decay_gammas * state[:, :, i - 1]  # b h d1 d2

    # For ease of using these states for our purposes, rearrange so that it is in the batch dim:
    state = rearrange(state, 'b h t d1 d2 -> t b h d1 d2')

    if prev_state is not None:
        # Update the retention Tensor, based on cross-chunk information
        inner_decay = decay_gammas ** inner_pos
        retention = retention + (
                einsum(query, prev_state, "b h n d1, b h d1 d2 -> b h n d2") * inner_decay
        )

    return retention, state


@torch.compile()
def _multiply_by_i(x: Tensor) -> Tensor:
    """Multiply a complex-valued tensor by the imaginary unit 'i'."""
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(start_dim=-2)


@torch.compile()
def _theta_shift(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # TODO: Add docstring
    return (x * cos) + (_multiply_by_i(x) * sin)


def retention_chunkwise(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        prev_state: Optional[Tensor],
        tokens_per_block: int,
        mask_type: str = 'autoregressive',
        scale: Optional[float] = None,
        compute_state: bool = True,
        decay_scale_min_num_blocks: int = DECAY_SCALE_MIN_NUM_BLOCKS,
        decay_scale_max_num_blocks: int = DECAY_SCALE_MAX_NUM_BLOCKS,
) -> Tuple[Tensor, Tensor]:
    decay_gammas = _build_decay_gammas(
        num_heads=query.shape[1], device=query.device, dtype=query.dtype,
        xmin=log(1 / (decay_scale_min_num_blocks * tokens_per_block)),
        xmax=log(1 / (decay_scale_max_num_blocks * tokens_per_block)),
    )
    decay_mask = _build_decay_mask(
        num_heads=query.shape[1],
        query_length=query.shape[2],
        key_length=key.shape[2],
        tokens_per_block=tokens_per_block,
        mask_type=mask_type,
        device=query.device,
        dtype=query.dtype,
        decay_scale_min_num_blocks=decay_scale_min_num_blocks,
        decay_scale_max_num_blocks=decay_scale_max_num_blocks
    )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - n / s: seq_length
    # - d: head_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    # intra-chunk (same as parallel retention)
    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
    retention = einsum(similarity, value, "b h n s, b h s d -> b h n d")

    # cross-chunk (derived from recurrent retention)
    decay_gammas = rearrange(decay_gammas, "h -> () h () ()")
    inner_pos = rearrange(
        torch.arange(key.size(2), device=key.device, dtype=key.dtype) + 1,
        "n -> () () n ()",
    )
    if compute_state:
        state_decays = decay_gammas ** (key.size(2) - inner_pos)
        discounted_key = einsum(key, state_decays, 'b h n d, _ h n _ -> b h n d')
        state = einsum(discounted_key, value, 'b h n d1, b h n d2 -> b h d1 d2')

        if prev_state is not None:
            # Update internal state to return to the user
            chunk_decay = decay_gammas ** key.size(2)
            state = state + prev_state * chunk_decay
    else:
        state = prev_state

    if prev_state is not None:
        # Update the retention Tensor, based on cross-chunk information
        inner_decay = decay_gammas ** inner_pos
        retention = retention + (
                einsum(query, prev_state, "b h n d1, b h d1 d2 -> b h n d2") * inner_decay
        )

    return retention, state


@torch.compile()
def apply_relative_position(q, k, start_idx: Union[int, torch.Tensor], thetas: Tensor) -> Tuple[Tensor, Tensor]:
    indices = torch.arange(q.size(2), device=q.device, dtype=q.dtype)

    if isinstance(start_idx, int):
        assert thetas is not None
        # Combined (cross + intra chunk):
        indices = start_idx + indices
        indices = indices.reshape(1, 1, -1, 1)

    elif isinstance(start_idx, torch.Tensor):
        assert start_idx.dim() == 1
        indices = start_idx.view(-1, 1) + indices.view(1, -1)
        indices = indices.reshape(indices.shape[0], 1, indices.shape[1], 1)

    else:
        assert False, f"Unsupported type for start_index. Expected int or LongTensor, got '{type(start_idx)}'."

    thetas = thetas.reshape(1, 1, 1, -1)
    angles = indices * thetas
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    q = _theta_shift(q, sin, cos)
    k = _theta_shift(k, sin, cos)

    return q, k


@torch.compile()
def apply_relative_position_pred_tokens(q, k, start_idx: Union[int, torch.Tensor], thetas: Tensor, tokens_per_block: int) -> Tuple[Tensor, Tensor]:
    assert q.dim() == 5 and k.dim() == 5
    indices = torch.arange(q.size(3), device=q.device, dtype=q.dtype).reshape(1, -1)
    block_steps = torch.arange(q.size(2), device=q.device, dtype=q.dtype).reshape(-1, 1) * tokens_per_block
    indices = indices + block_steps
    indices = indices.flatten()
    # b h t k d -> b h (t k) d
    q = q.flatten(2, 3)
    k = k.flatten(2, 3)


    if isinstance(start_idx, int):
        assert thetas is not None
        # Combined (cross + intra chunk):
        indices = start_idx + indices
        indices = indices.reshape(1, 1, -1, 1)

    elif isinstance(start_idx, torch.Tensor):
        assert start_idx.dim() == 1
        indices = start_idx.view(-1, 1) + indices.view(1, -1)
        indices = indices.reshape(indices.shape[0], 1, indices.shape[1], 1)

    else:
        assert False, f"Unsupported type for start_index. Expected int or LongTensor, got '{type(start_idx)}'."

    thetas = thetas.reshape(1, 1, 1, -1)
    angles = indices * thetas
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    q = _theta_shift(q, sin, cos)
    k = _theta_shift(k, sin, cos)

    return q, k


class POPMultiScaleRetention(yet_another_retnet.retention.MultiScaleRetention):

    def __init__(self, embed_dim: int, num_heads: int, tokens_per_block: int, dropout: float = 0.0,
                 relative_position: bool = True,
                 bias: bool = True, batch_first: bool = True,
                 activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
                 group_norm_eps: float = 1e-6, device: Optional[Union[torch.device, str]] = None,
                 dtype: Optional[torch.dtype] = None, mask_type: str = 'autoregressive',
                 decay_scale_min_num_blocks: int = DECAY_SCALE_MIN_NUM_BLOCKS,
                 decay_scale_max_num_blocks: int = DECAY_SCALE_MAX_NUM_BLOCKS,
                 ):
        super().__init__(embed_dim, num_heads, dropout, relative_position, bias, batch_first, activation,
                         group_norm_eps, device, dtype)
        self.decay_scale_min_num_blocks = decay_scale_min_num_blocks
        self.decay_scale_max_num_blocks = decay_scale_max_num_blocks
        self.mask_type = mask_type
        self.tokens_per_block = tokens_per_block

        # logger.info(f"Using decay scale range [{decay_scale_min_num_blocks}, {decay_scale_max_num_blocks}] (blocks)")

    def retention_chunkwise_per_block_states(
            self,
            x: Tensor,
            start_idx: Union[int, torch.LongTensor],
            prev_state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # einstein notation:
        # b - batch size
        # n - sequence length
        # h - number of heads
        # d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(x)
        k: Tensor = self.k_proj(x)
        v: Tensor = self.v_proj(x)

        # Unfold 'd' dimension into 'h' separate retention heads.  Move the head
        # dimension to position 1 (makes matrix ops *much* faster).
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        if self.relative_position:
            assert self.thetas is not None
            # Combined (cross + intra chunk):
            q, k = apply_relative_position(q, k, start_idx, self.thetas)

        # Apply retention then group norm.
        retention, state = retention_chunkwise_per_block_states(
            q, k, v,
            prev_state=prev_state,
            tokens_per_block=self.tokens_per_block,
            mask_type=self.mask_type,
            decay_scale_min_num_blocks=self.decay_scale_min_num_blocks,
            decay_scale_max_num_blocks=self.decay_scale_max_num_blocks
        )
        # To apply group norm in an equivalent way to the recurrent formulation,
        # we fold the sequence dimension into the batch dimension.  Otherwise,
        # normalization would be applied over the entire input sequence.
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) (h d)")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) e -> b n e", b=batch_size)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'query' can equivalently be used as the input.
        gate = self.activation(self.g_proj(x))
        retention = self.out_proj(retention * gate)

        return retention, state

    def forward_chunkwise_from_per_block_states(
            self,
            x_pred_tokens: Tensor,
            start_idx: Union[int, torch.LongTensor],
            prev_states_per_block: Optional[Tensor],
            compute_state: bool = True
    ) -> Tuple[Tensor, Tensor]:
        tokens_per_block = self.tokens_per_block

        # einstein notation:
        # b - batch size
        # n - sequence length
        # h - number of heads
        # d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(x_pred_tokens)
        k: Tensor = self.k_proj(x_pred_tokens)
        v: Tensor = self.v_proj(x_pred_tokens)

        # Unfold 'd' dimension into 'h' separate retention heads.  Move the head
        # dimension to position 1 (makes matrix ops *much* faster).
        # here I assume that the pred tokens are already arranged s.t. n = (t k) -> t, k
        q = rearrange(q, "b t k (h d) -> b h t k d", h=self.num_heads)
        k = rearrange(k, "b t k (h d) -> b h t k d", h=self.num_heads)
        v = rearrange(v, "b t k (h d) -> (t b) h k d", h=self.num_heads)
        tokens_per_obs = q.shape[3]

        if self.relative_position:
            assert self.thetas is not None
            # Combined (cross + intra chunk):
            q, k = apply_relative_position_pred_tokens(q, k, start_idx, self.thetas, self.tokens_per_block)

        # convert to a batch of time steps:
        q = rearrange(q, 'b h (t k) d -> (t b) h k d', k=tokens_per_obs)
        k = rearrange(k, 'b h (t k) d -> (t b) h k d', k=tokens_per_obs)

        # shift the states:
        t = prev_states_per_block.shape[0]
        prev_states_per_block = rearrange(prev_states_per_block, 't b h d1 d2 -> (t b) h d1 d2')

        # Apply retention then group norm.
        retention, state = retention_chunkwise(
            q, k, v,
            tokens_per_block=tokens_per_block,
            mask_type=self.mask_type,
            prev_state=prev_states_per_block,
            compute_state=compute_state,
            decay_scale_min_num_blocks=self.decay_scale_min_num_blocks,
            decay_scale_max_num_blocks=self.decay_scale_max_num_blocks
        )
        retention = rearrange(retention, '(t b) h k d -> b h (t k) d', k=tokens_per_obs, t=t)
        # To apply group norm in an equivalent way to the recurrent formulation,
        # we fold the sequence dimension into the batch dimension.  Otherwise,
        # normalization would be applied over the entire input sequence.
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) (h d)")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) e -> b n e", b=batch_size)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'query' can equivalently be used as the input.
        gate = self.activation(self.g_proj(x_pred_tokens))
        retention = rearrange(retention, 'b (t k) e -> b t k e', t=t, k=tokens_per_obs)
        retention = self.out_proj(retention * gate)

        return retention, state

    def forward_chunkwise(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            start_idx: Union[int, torch.LongTensor],
            prev_state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # einstein notation:
        # b - batch size
        # n - sequence length
        # h - number of heads
        # d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate retention heads.  Move the head
        # dimension to position 1 (makes matrix ops *much* faster).
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        if self.relative_position:
            # global (cross-chunk) + intra-chunk relative position embedding
            assert self.thetas is not None
            q, k = apply_relative_position(q, k, start_idx, self.thetas)

        # Apply retention then group norm.
        retention, state = retention_chunkwise(
            q, k, v,
            tokens_per_block=self.tokens_per_block,
            mask_type=self.mask_type,
            prev_state=prev_state,
            decay_scale_min_num_blocks=self.decay_scale_min_num_blocks,
            decay_scale_max_num_blocks=self.decay_scale_max_num_blocks
        )
        # To apply group norm in an equivalent way to the recurrent formulation,
        # we fold the sequence dimension into the batch dimension.  Otherwise,
        # normalization would be applied over the entire input sequence.
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) (h d)")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) e -> b n e", b=batch_size)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'query' can equivalently be used as the input.
        gate = self.activation(self.g_proj(query))
        retention = self.out_proj(retention * gate)

        return retention, state


def _pop_retention_block(retention, x: Tensor, x_pred_tokens: Tensor, start_idx, prev_state, dropout) -> Tuple[Tensor, Tensor, Tensor]:
    x, state = retention.retention_chunkwise_per_block_states(
        x, start_idx=start_idx, prev_state=prev_state
    )

    # shift the states so that o_t is predicted from s_{t-1} :
    if prev_state is None:
        s0 = torch.zeros_like(state[0:1])
        shifted_states = torch.cat([s0, state[:-1]], dim=0)
    else:
        shifted_states = torch.cat([prev_state.unsqueeze(0), state[:-1]], dim=0)

    x_pred_tokens, _ = retention.forward_chunkwise_from_per_block_states(
        x_pred_tokens, start_idx=start_idx, prev_states_per_block=shifted_states, compute_state=False
    )
    return dropout(x), dropout(x_pred_tokens), state[-1]


class POPRetNetDecoderLayer(yet_another_retnet.retnet.RetNetDecoderLayer):

    def __init__(self, d_model: int, nhead: int, tokens_per_block: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish", norm_first: bool = True,
                 layer_norm_eps: float = 1e-6, device: Optional[Union[torch.device, str]] = None,
                 dtype: Optional[torch.dtype] = None, mask_type: str = 'autoregressive',
                 decay_scale_min_num_blocks: int = DECAY_SCALE_MIN_NUM_BLOCKS,
                 decay_scale_max_num_blocks: int = DECAY_SCALE_MAX_NUM_BLOCKS, 
                 token_feedforward_alone: bool = False ) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, norm_first, layer_norm_eps, device,
                         dtype)
        if token_feedforward_alone:
            self.tokenlinear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
            self.tokenlinear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)
        self._token_feedforward_alone = token_feedforward_alone
        self.retention = POPMultiScaleRetention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            tokens_per_block=tokens_per_block,
            dropout=dropout,
            activation=activation,
            device=device,
            dtype=dtype,
            mask_type=mask_type,
            decay_scale_min_num_blocks=decay_scale_min_num_blocks,
            decay_scale_max_num_blocks=decay_scale_max_num_blocks
        )
    def _tokenfeedforward_block(self, x: Tensor) -> Tensor:
        if self._token_feedforward_alone:
            x = self.activation(self.tokenlinear1(x))
        else:
            x = self.activation(self.linear1(x))
        x = self.dropout(x)
        if self._token_feedforward_alone:
            x = self.tokenlinear2(x)
        else:
            x = self.linear2(x)
        x = self.dropout(x)
        return x
    def pop_forward(
            self, x: Tensor, x_pred_tokens: Tensor, start_idx: Union[int, torch.LongTensor], prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # retention block
        if self.norm_first:
            y_x, y_pred_tokens, state = _pop_retention_block(
                self.retention, self.norm1(x), self.norm1(x_pred_tokens), start_idx, prev_state, dropout=self.dropout)
            x = x + y_x
            x_pred_tokens = x_pred_tokens + y_pred_tokens
            x = x + self._feedforward_block(self.norm2(x))
            x_pred_tokens = x_pred_tokens + self._tokenfeedforward_block(self.norm2(x_pred_tokens))
        else:
            y_x, y_pred_tokens, state = _pop_retention_block(self.retention, x, x_pred_tokens, start_idx, prev_state, self.dropout)
            x = x + self.norm1(y_x)
            x_pred_tokens = x_pred_tokens + self.norm1(y_pred_tokens)
            x = x + self.norm2(self._feedforward_block(x))
            x_pred_tokens = x_pred_tokens + self.norm2(self._tokenfeedforward_block(x_pred_tokens)) #fix org error:x_pred_tokens = x_pred_tokens + self.norm2(x_pred_tokens)

        return x, x_pred_tokens, state


class POPRetNetDecoder(yet_another_retnet.retnet.RetNetDecoder):

    def __init__(self, decoder_layers: list[POPRetNetDecoderLayer]):
        super().__init__(None, 0)
        self.num_layers = len(decoder_layers)
        self.layers = nn.ModuleList(decoder_layers)

    def pop_forward(
            self, x: Tensor, x_pred_tokens: Tensor, start_idx: Union[int, torch.LongTensor], prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if prev_states is None or len(prev_states) == 0:
            prev_states = [None] * self.num_layers
        elif len(prev_states) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} previous states, got {len(prev_states)}"
            )

        states: List[Tensor] = []
        for layer, prev_state in zip(self.layers, prev_states):
            assert isinstance(layer, POPRetNetDecoderLayer)
            x, x_pred_tokens, state = layer.pop_forward(x, x_pred_tokens, start_idx, prev_state)
            states.append(state)
        return x, x_pred_tokens, torch.stack(states)

    def forward_chunkwise(
            self, x: Tensor, start_idx: Union[int, torch.LongTensor], prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, Tensor]:
        if prev_states is None or len(prev_states) == 0:
            prev_states = [None] * self.num_layers
        elif len(prev_states) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} previous states, got {len(prev_states)}"
            )

        states: List[Tensor] = []
        for layer, prev_state in zip(self.layers, prev_states):
            assert isinstance(layer, POPRetNetDecoderLayer)
            x, state = layer.forward_chunkwise(x, start_idx, prev_state)
            states.append(state)
        return x, torch.stack(states)


def test_per_frame_chunkwise_states():
    DEVICE = "cuda:0"
    DTYPE = torch.float32
    batch_size, num_heads, seq_length, hidden_dim = 5, 4, 10 * 65, 256

    size = (batch_size, num_heads, seq_length, hidden_dim)
    query = torch.randn(*size, device=DEVICE, dtype=DTYPE)
    key = torch.randn(*size, device=DEVICE, dtype=DTYPE)
    value = torch.randn(*size, device=DEVICE, dtype=DTYPE)

    r1, s1 = retention_chunkwise_per_block_states(query, key, value, prev_state=None, tokens_per_block=64)

    r2, s2 = yar.retention.retention_chunkwise(query, key, value, prev_state=None)

    torch.testing.assert_close(s1[-1], s2)
    torch.testing.assert_close(r1, r2)


if __name__ == '__main__':
    test_per_frame_chunkwise_states()

