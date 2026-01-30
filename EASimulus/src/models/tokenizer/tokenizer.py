"""
Credits to https://github.com/CompVis/taming-transformers
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, Optional

import einops
from loguru import logger
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import FreqDist
from torch import Tensor

from dataset import Batch
from .lpips import LPIPS
from .nets import SimpleEncoder, SimpleDecoder
from utils import LossWithIntermediateLosses, VectorQuantizer, ObsModality
from utils.math import base_n_to_base_10, base_10_to_base_n


@dataclass
class TokenizerEncoderOutput:
    z: Tensor
    z_quantized: Tensor
    tokens: Tensor


class TokenizerBase(ABC, nn.Module):
    @property
    @abstractmethod
    def modality(self) -> ObsModality:
        pass

    @property
    @abstractmethod
    def is_trainable(self) -> bool:
        pass

    @property
    @abstractmethod
    def tokens_per_obs(self) -> int:
        pass

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
        return_tokens: bool = False,
    ) -> Tuple[Tensor, ...]:
        pass

    @abstractmethod
    def compute_loss(
        self, batch: Batch, **kwargs: Any
    ) -> tuple[LossWithIntermediateLosses, dict]:
        pass

    @abstractmethod
    def encode(
        self, x: Tensor, should_preprocess: bool = False
    ) -> TokenizerEncoderOutput:
        pass

    @abstractmethod
    def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
        pass

    @abstractmethod
    def to_codes(self, tokens, **kwargs):
        pass

    @torch.no_grad()
    def encode_decode(
        self,
        x: Tensor,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
    ) -> Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)


def _combine_encoder_outputs(
    outputs: list[TokenizerEncoderOutput], input_shape=None
) -> TokenizerEncoderOutput:
    assert len(outputs) > 0
    results = (
        TokenizerEncoderOutput(
            z=torch.cat([r_i.z for r_i in outputs], dim=0),
            z_quantized=torch.cat([r_i.z_quantized for r_i in outputs], dim=0),
            tokens=torch.cat([r_i.tokens for r_i in outputs], dim=0),
        )
        if len(outputs) > 1
        else outputs[0]
    )
    if input_shape is not None:
        results.z = results.z.reshape(*input_shape[:-3], *results.z.shape[1:])
        results.z_quantized = results.z_quantized.reshape(
            *input_shape[:-3], *results.z_quantized.shape[1:]
        )
        results.tokens = results.tokens.reshape(
            *input_shape[:-3], *results.tokens.shape[1:]
        )
    return results


class ImageTokenizer(TokenizerBase):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        vgg_lpips_ckpt_path: str,
        encoder: SimpleEncoder,
        decoder: SimpleDecoder,
        with_lpips: bool = True,
        lpips_pnet='vgg16',
        device=None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder.to(device)
        self.pre_quant_conv = (
            nn.Identity()
        )  # torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        self.embedding = nn.Embedding(vocab_size, embed_dim, device=device)
        self.post_quant_conv = (
            nn.Identity()
        )  # torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder.to(device)
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        if with_lpips:
            self.lpips=LPIPS(vgg_lpips_ckpt_path,True,lpips_pnet).eval().to(device) 
        else:
            self.lpips=None

        self._effective_bsz = None
        self._past_err_msgs = []

    def __repr__(self) -> str:
        return "tokenizer"

    @property
    def modality(self) -> ObsModality:
        return ObsModality.image

    @property
    def is_trainable(self) -> bool:
        return True

    @property
    def tokens_per_obs(self) -> int:
        res = np.array(self.encoder.config.input_resolution)
        return int(np.prod(res / 2**self.encoder.config.num_downsample_steps))

    def forward(
        self,
        x: Tensor,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
        return_tokens: bool = False,
    ) -> Tuple[Tensor, ...]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        if return_tokens:
            return outputs.z, outputs.z_quantized, reconstructions, outputs.tokens
        return outputs.z, outputs.z_quantized, reconstructions

    def _auto_adjust_bsz_call(self, x: Tensor, fn, combine_results_fn, **kwargs):
        """
        Automatically adjust the effective batch size and call `fn` with apropriate input sized batch.
        in case the input is too large, split the computation into multiple (sequential) calls
        of smaller size.
        """
        input_shape = x.shape
        x = x.reshape(-1, *input_shape[-3:])
        input_bsz = x.shape[0]
        bsz = input_bsz if self._effective_bsz is None else self._effective_bsz
        while bsz > 0:
            try:
                num_mini_batches = math.ceil(input_bsz / bsz)
                results = [
                    fn(x[i * bsz : (i + 1) * bsz], **kwargs)
                    for i in range(num_mini_batches)
                ]

                results = combine_results_fn(results, input_shape)
                if bsz < input_bsz and self._effective_bsz is None:
                    self._effective_bsz = bsz
                return results
            except torch.OutOfMemoryError:
                err_msg = f"Out of Memory with batch size = {bsz}, trying {bsz // 2}..."
                if err_msg not in self._past_err_msgs:
                    self._past_err_msgs.append(err_msg)
                    logger.warning(err_msg)
                bsz = bsz // 2

        raise RuntimeError("No batch size fits the available memory!")

    def compute_loss(
        self, batch: Batch, **kwargs: Any
    ) -> tuple[LossWithIntermediateLosses, dict]:
        # assert self.lpips is not None
        obs = batch["observations"][ObsModality.image]
        t = obs.shape[1]
        b = obs.shape[0]
        assert t == 1
        observations = self.preprocess_input(rearrange(obs, "b t c h w -> (b t) c h w"))
        z, z_quantized, reconstructions, tokens = self(
            observations,
            should_preprocess=False,
            should_postprocess=False,
            return_tokens=True,
        )

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        z = z.reshape(b, -1)
        z_quantized = z_quantized.reshape(b, -1)
        commitment_loss = F.mse_loss(z_quantized, z.detach()) + beta * F.mse_loss(
            z, z_quantized.detach()
        )

        if self.lpips is not None:
            perceptual_loss = self.lpips(observations, reconstructions).flatten()
            perceptual_loss = torch.mean(perceptual_loss)
        else:
            perceptual_loss = torch.zeros_like(commitment_loss)

        reconstruction_loss = F.mse_loss(observations, reconstructions)

        with torch.no_grad():
            info = {
                "codebook_norms": torch.norm(self.embedding.weight, dim=1),
                "token_counts": FreqDist(tokens.detach().flatten().cpu().numpy()),
                # 'per_sample_loss': per_sample_loss
            }

        return (
            LossWithIntermediateLosses(
                commitment_loss=commitment_loss,
                reconstruction_loss=reconstruction_loss,
                perceptual_loss=perceptual_loss,
            ),
            info,
        )

    def encode(
        self, x: Tensor, should_preprocess: bool = False
    ) -> TokenizerEncoderOutput:
        return self._auto_adjust_bsz_call(
            x,
            self._encode,
            _combine_encoder_outputs,
            should_preprocess=should_preprocess,
        )

    def _encode(
        self, x: Tensor, should_preprocess: bool = False
    ) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        b, e, h, w = z.shape
        z_flattened = rearrange(z, "b e h w -> (b h w) e")
        dist_to_embeddings = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = rearrange(
            self.embedding(tokens), "(b h w) e -> b e h w", b=b, e=e, h=h, w=w
        ).contiguous()

        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.reshape(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec

    @torch.no_grad()
    def to_codes(self, tokens: Tensor, **kwargs):
        hw = tokens.shape[-1]
        h = int(
            np.sqrt(
                hw
                * self.encoder.config.input_resolution[0]
                / self.encoder.config.input_resolution[1]
            )
        )
        w = int(hw / h)
        emb = self.embedding(tokens)  # (B * hw, e)
        z_q = rearrange(emb, "... (h w) e -> ... e h w", h=h, w=w).contiguous()
        return z_q

    @torch.no_grad()
    def encode_decode(
        self,
        x: Tensor,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
    ) -> Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: Tensor) -> Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: Tensor) -> Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)


class HardCodedVectorTokenizer(TokenizerBase):

    def __init__(
        self,
        input_dim: int,
        vector_quantizer: VectorQuantizer = None,
        device=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.vector_quantizer = (
            VectorQuantizer(normalize=False)
            if vector_quantizer is None
            else vector_quantizer
        )
        # self.vector_quantizer = ByteToken2FP16Mapper()
        if device is not None:
            self.vector_quantizer.to(device)

    @property
    def modality(self) -> ObsModality:
        return ObsModality.vector

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def tokens_per_obs(self) -> int:
        return self.input_dim

    @property
    def vocab_size(self):
        return self.vector_quantizer.vocab_size

    def forward(
        self,
        x: Tensor,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
        return_tokens: bool = False,
    ) -> Tuple[Tensor, ...]:
        outputs = self.encode(x, should_preprocess)
        reconstructions = outputs.z_quantized
        if return_tokens:
            return outputs.z, outputs.z_quantized, reconstructions, outputs.tokens
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(
        self, batch: Batch, **kwargs: Any
    ) -> tuple[LossWithIntermediateLosses, dict]:
        return LossWithIntermediateLosses(), {}

    def encode(
        self, x: Tensor, should_preprocess: bool = False
    ) -> TokenizerEncoderOutput:
        z = x
        tokens = self.vector_quantizer.vector_to_tokens_pt(x)
        z_q = self.vector_quantizer.tokens_to_vector_pt(tokens)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
        return z_q

    def to_codes(self, tokens, **kwargs):
        return self.vector_quantizer.tokens_to_vector_pt(tokens)


class VectorTokenizer(TokenizerBase):

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        vocab_size: int,
        device=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self._tokens_per_obs = 32

        self.codebook = nn.Embedding(
            vocab_size * self.tokens_per_obs, embed_dim, device=device
        )
        self.codebook.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

        self.encoder = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim * 2, device=device),
            nn.ReLU(),
            # nn.Linear(embedding_dim * 2, embedding_dim * 2),
            # nn.ReLU(),
            # nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, vocab_size * self.tokens_per_obs, device=device),
        )
        self.decoder = nn.Sequential(
            # nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2, device=device),
            nn.ReLU(),
            # nn.Linear(embedding_dim * 2, embedding_dim * 2),
            # nn.ReLU(),
            # nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, input_dim, device=device),
        )

    def __repr__(self):
        return "VectorTokenizer"

    @property
    def modality(self) -> ObsModality:
        return ObsModality.vector

    @property
    def is_trainable(self) -> bool:
        return True

    @property
    def tokens_per_obs(self) -> int:
        return self._tokens_per_obs

    def forward(
        self,
        x: Tensor,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
        return_tokens: bool = False,
    ) -> Tuple[Tensor, ...]:
        outputs = self.encode(x, should_preprocess)
        # decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        decoder_input = outputs.z_quantized
        reconstructions = self.decode(decoder_input, should_postprocess)
        if return_tokens:
            return outputs.z, outputs.z_quantized, reconstructions, outputs.tokens
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(
        self, batch: Batch, **kwargs: Any
    ) -> tuple[LossWithIntermediateLosses, dict]:
        t = batch["observations"].shape[1]
        b = batch["observations"].shape[0]
        assert t == 1
        observations = batch["observations"]
        z, z_quantized, reconstructions, tokens = self(
            observations,
            should_preprocess=False,
            should_postprocess=False,
            return_tokens=True,
        )

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        # z = z.reshape(b, -1)
        # z_quantized = z_quantized.reshape(b, -1)
        # commitment_loss = F.mse_loss(z_quantized, z.detach()) + beta * F.mse_loss(z, z_quantized.detach())

        reconstruction_loss = F.mse_loss(observations, reconstructions)

        with torch.no_grad():
            info = {
                "codebook_norms": torch.norm(self.codebook.weight, dim=1),
                "token_counts": FreqDist(tokens.detach().flatten().cpu().numpy()),
                # 'per_sample_loss': per_sample_loss
            }

        return LossWithIntermediateLosses(reconstruction_loss=reconstruction_loss), info

    def encode(
        self, x: Tensor, should_preprocess: bool = False
    ) -> TokenizerEncoderOutput:
        z = self.encoder(x.view(-1, x.shape[-1]))  # (b, k * vocab_size)
        z = rearrange(z, "b (k v) -> b k v", v=self.vocab_size, k=self.tokens_per_obs)
        tokens = torch.distributions.Categorical(logits=z).sample()
        one_hots = F.one_hot(tokens, self.vocab_size).flatten(-2)
        p = torch.softmax(z, dim=-1).flatten(-2)
        st_estimator = p + (one_hots - p).detach()

        z_q = st_estimator @ self.codebook.weight

        z_q = z_q.reshape(*x.shape[:-1], -1).contiguous()

        return TokenizerEncoderOutput(z.reshape(*x.shape[:-1], -1), z_q, tokens)

    def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
        rec = self.decoder(z_q)
        return rec

    def to_codes(self, tokens, **kwargs):
        return self.codebook(tokens)


class VQVectorTokenizerOld(TokenizerBase):

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        vocab_size: int,
        device=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.features_per_token = 3
        self._tokens_per_obs = int(np.ceil(input_dim / self.features_per_token))

        self.codebook = nn.Embedding(vocab_size, self.features_per_token, device=device)
        # self.codebook.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

        self.encoder = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.Linear(self.features_per_token, embed_dim, device=device),
            # nn.ReLU(),
            # nn.Linear(embedding_dim * 2, embedding_dim * 2),
            # nn.ReLU(),
            # nn.LayerNorm(embed_dim * 2),
            # nn.Linear(embed_dim * 2, embed_dim),
        )
        # self.encoder = nn.Sequential(
        #     nn.LayerNorm(input_dim),
        #     nn.Linear(input_dim, 64),
        #     nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.LayerNorm(7),
        #     nn.Conv1d(64, embed_dim, kernel_size=7, stride=2, padding=0),
        #     # nn.Linear(embedding_dim * 2, embedding_dim * 2),
        #     # nn.ReLU(),
        #     # nn.LayerNorm(embed_dim * 2),
        #     # nn.Linear(embed_dim * 2, embed_dim),
        # )
        self.decoder = nn.Sequential(
            # nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.SiLU(),
            # nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, self.features_per_token, device=device),
        )

        self.code_map = nn.Sequential(
            nn.Linear(self.features_per_token, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, self.features_per_token),
        )

    def __repr__(self):
        return "VectorTokenizer"

    @property
    def modality(self) -> ObsModality:
        return ObsModality.vector

    @property
    def is_trainable(self) -> bool:
        return True

    @property
    def tokens_per_obs(self) -> int:
        return self._tokens_per_obs

    def forward(
        self,
        x: Tensor,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
        return_tokens: bool = False,
    ) -> Tuple[Tensor, ...]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        if return_tokens:
            return outputs.z, outputs.z_quantized, reconstructions, outputs.tokens
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(
        self, batch: Batch, **kwargs: Any
    ) -> tuple[LossWithIntermediateLosses, dict]:
        t = batch["observations"].shape[1]
        b = batch["observations"].shape[0]
        assert t == 1
        observations = batch["observations"]
        z, z_quantized, reconstructions, tokens = self(
            observations,
            should_preprocess=False,
            should_postprocess=False,
            return_tokens=True,
        )

        # loguru.logger.debug(f"diff: {observations[0] - reconstructions[0]}")

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        # z = z.reshape(b, -1)
        # z_quantized = z_quantized.reshape(b, -1)
        commitment_loss = F.mse_loss(z_quantized, z.detach()) + beta * F.mse_loss(
            z, z_quantized.detach()
        )

        reconstruction_loss = F.mse_loss(observations, reconstructions)

        with torch.no_grad():
            info = {
                "codebook_norms": torch.norm(self.codebook.weight, dim=1),
                "token_counts": FreqDist(tokens.detach().flatten().cpu().numpy()),
                # 'per_sample_loss': per_sample_loss
            }

        return (
            LossWithIntermediateLosses(
                commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss
            ),
            info,
        )

    def encode(
        self, x: Tensor, should_preprocess: bool = False
    ) -> TokenizerEncoderOutput:
        # x = sym_log(x)
        k = self.tokens_per_obs
        d = self.features_per_token
        e = self.embed_dim
        residue = x.shape[-1] % self.features_per_token
        if residue > 0:
            # x = torch.cat([x[..., :-residue], x[..., -self.features_per_token:]], dim=-1)
            x = F.pad(
                x, (0, self.features_per_token - residue), mode="constant", value=0
            )
        x_flat = rearrange(x.view(-1, x.shape[-1]), "b (k d) -> b k d", d=d, k=k)
        z = self.encoder(x_flat)
        latent_codes = self.code_map(self.codebook.weight)  # (v d)
        dist_to_embeddings = (
            torch.sum(x_flat**2, dim=-1, keepdim=True)
            + torch.sum(latent_codes**2, dim=-1).view(1, 1, -1)
            - 2 * torch.matmul(x_flat, latent_codes.t())
        )

        tokens = dist_to_embeddings.argmin(dim=-1).long().reshape(*x.shape[:-1], -1)
        z_q = (
            self.encoder(self.code_map(self.codebook(tokens)))
            .reshape(*x.shape[:-1], k, e)
            .contiguous()
        )

        return TokenizerEncoderOutput(z.reshape(*x.shape[:-1], k, e), z_q, tokens)

    def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
        # z_q = rearrange(z_q, '... (k d) -> ... k d', k=self.tokens_per_obs, d=self.features_per_token)
        rec = self.decoder(z_q).flatten(-2)
        return rec

    def to_codes(self, tokens, **kwargs):
        return self.encoder(self.code_map(self.codebook(tokens)))


class VQVectorTokenizer(TokenizerBase):

    def __init__(
        self,
        input_dim: int,
        bits_per_code: int = 12,
        hidden_dim: int = 256,
        device=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = 2**bits_per_code

        self.bits_per_code = bits_per_code
        self._tokens_per_obs = 1

        self.encoder = nn.Sequential(
            # nn.LayerNorm(input_dim, device=device),
            nn.Linear(input_dim, hidden_dim, device=device),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim, device=device),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim, device=device),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim, device=device),
            nn.Linear(hidden_dim, bits_per_code, device=device),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bits_per_code, hidden_dim, device=device),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim, device=device),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim, device=device),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim, device=device),
            nn.Linear(hidden_dim, input_dim, device=device),
        )

    def __repr__(self):
        return "VectorTokenizer"

    @property
    def modality(self) -> ObsModality:
        return ObsModality.vector

    @property
    def is_trainable(self) -> bool:
        return True

    @property
    def tokens_per_obs(self) -> int:
        return self._tokens_per_obs

    def forward(
        self,
        x: Tensor,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
        return_tokens: bool = False,
    ) -> Tuple[Tensor, ...]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        if return_tokens:
            return outputs.z, outputs.z_quantized, reconstructions, outputs.tokens
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(
        self, batch: Batch, **kwargs: Any
    ) -> tuple[LossWithIntermediateLosses, dict]:
        observations = batch["observations"][ObsModality.vector]
        assert observations.shape[1] == 1
        z, z_quantized, reconstructions = self(
            observations, should_preprocess=False, should_postprocess=False
        )

        commitment_loss = F.mse_loss(z, z_quantized.detach())

        reconstruction_loss = F.mse_loss(observations, reconstructions)
        logger.debug(f"{observations[0]}; {reconstructions[0]}")

        info = {}

        return (
            LossWithIntermediateLosses(
                commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss
            ),
            info,
        )

    def encode(
        self, x: Tensor, should_preprocess: bool = False
    ) -> TokenizerEncoderOutput:
        k = self.bits_per_code
        z = self.encoder(x)
        z_q = torch.where(z > 0, torch.ones_like(z), torch.zeros_like(z))
        tokens = base_n_to_base_10(z_q, 2, num_digits=k)

        return TokenizerEncoderOutput(z, z, tokens)

    def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
        rec = self.decoder(z_q)
        return rec

    def to_codes(self, tokens, **kwargs):
        return base_10_to_base_n(tokens, 2, num_digits=self.bits_per_code)


class DummyTokenizer(TokenizerBase):

    def __init__(self, nvec: np.ndarray, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nvec = nvec
        if nvec.ndim == 1:
            self._modality = ObsModality.token
        else:
            assert nvec.ndim == 2, f"{nvec.ndim}-dim is not supported"
            self._modality = ObsModality.token_2d

    @property
    def modality(self) -> ObsModality:
        return self._modality

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def tokens_per_obs(self) -> int:
        return self.nvec.shape[0]

    @property
    def vocab_size(self):
        return self.nvec[0]

    def forward(
        self,
        x: Tensor,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
        return_tokens: bool = False,
    ) -> Tuple[Tensor, ...]:
        return x, x, x

    def compute_loss(
        self, batch: Batch, **kwargs: Any
    ) -> tuple[LossWithIntermediateLosses, dict]:
        return LossWithIntermediateLosses(), {}

    def encode(
        self, x: Tensor, should_preprocess: bool = False
    ) -> TokenizerEncoderOutput:
        return TokenizerEncoderOutput(x, x, x)

    def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
        return z_q

    def to_codes(self, tokens, **kwargs):
        return tokens

    def encode_decode(
        self,
        x: Tensor,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
    ) -> Tensor:
        return super().encode_decode(x, should_preprocess, should_postprocess)


class MultiModalTokenizer(nn.Module):
    def __init__(self, tokenizers: dict[ObsModality, TokenizerBase], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizers = nn.ModuleDict({k.name: v for k, v in tokenizers.items()})

    def __repr__(self):
        return "tokenizers"

    @property
    def modalities(self) -> set[ObsModality]:
        return set([t.modality for t in self.tokenizers.values()])

    @property
    def is_trainable(self) -> bool:
        return any([t.is_trainable for t in self.tokenizers.values()])

    @property
    def tokens_per_obs(self) -> int:
        return sum([t.tokens_per_obs for t in self.tokenizers.values()])

    @property
    def tokens_per_obs_dict(self) -> dict[ObsModality, int]:
        return {ObsModality[k]: v.tokens_per_obs for k, v in self.tokenizers.items()}

    @property
    def vocab_size(self) -> dict[ObsModality, int]:
        return {ObsModality[k]: v.vocab_size for k, v in self.tokenizers.items()}

    def forward(
        self,
        x: dict[ObsModality, Tensor],
        should_preprocess: bool = False,
        should_postprocess: bool = False,
        return_tokens: bool = False,
    ) -> dict[ObsModality, Tuple[Tensor]]:
        assert set(x.keys()) == set(
            [ObsModality[k] for k in self.tokenizers.keys()]
        ), f"Obs keys ({x.keys()}) != tokenizers keys ({self.tokenizers.keys()})"
        return {
            ObsModality[k]: self.tokenizers[k].forward(
                x[ObsModality[k]], should_preprocess, should_postprocess, return_tokens
            )
            for k in self.tokenizers.keys()
        }

    def compute_loss(
        self, batch: Batch, **kwargs: Any
    ) -> tuple[LossWithIntermediateLosses, dict]:
        losses = {
            ObsModality[k]: self.tokenizers[k].compute_loss(batch, **kwargs)
            for k in self.tokenizers.keys()
        }
        combined = LossWithIntermediateLosses.combine([l[0] for l in losses.values()])
        infos = {k.name: l[1] for k, l in losses.items()}

        return combined, infos

    def encode(
        self, x: dict[ObsModality, Tensor], should_preprocess: bool = False
    ) -> dict[ObsModality, TokenizerEncoderOutput]:
        # if set(x.keys()) != set(
        #     [ObsModality[k] for k in self.tokenizers.keys()]):
        #     print(f"Obs keys ({x.keys()}) != tokenizers keys ({self.tokenizers.keys()})")
        return {
            ObsModality[k]: self.tokenizers[k].encode(
                x[ObsModality[k]], should_preprocess
            )
            for k in self.tokenizers.keys()
        }

    def decode(
        self, z_q: dict[ObsModality, Tensor], should_postprocess: bool = False
    ) -> dict[ObsModality, Tensor]:
        return {
            ObsModality[k]: self.tokenizers[k].decode(
                z_q[ObsModality[k]], should_postprocess
            )
            for k in self.tokenizers.keys()
        }

    def to_codes(
        self, tokens: dict[ObsModality, Tensor], **kwargs
    ) -> dict[ObsModality, Tensor]:
        return {
            ObsModality[k]: self.tokenizers[k].to_codes(
                tokens[ObsModality[k]], **kwargs
            )
            for k in self.tokenizers.keys()
        }

    def encode_decode(
        self,
        x: dict[ObsModality, Tensor],
        should_preprocess: bool = False,
        should_postprocess: bool = False,
    ) -> dict[ObsModality, Tensor]:
        encoded = self.encode(x, should_preprocess=should_preprocess)
        return self.decode(
            {k: v.z_quantized for k, v in encoded.items()},
            should_postprocess=should_postprocess,
        )
