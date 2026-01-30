import re
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Optional, Union

import cv2
import random
import shutil

import imageio
from einops import rearrange, einsum
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import wandb
from loguru import logger
from nltk import FreqDist
from torch import Tensor
import torch.nn.functional as F

from utils.math import sym_log, sym_exp
from utils.types import ObsModality


def configure_optimizer(model, learning_rate, weight_decay):
    decay = set()
    no_decay = set()

    for pn, p in model.named_parameters():
        param_names_lst = pn.split(sep='.')
        layer_norm_pattern = '\.ln(\d+)|(_f)\.'

        if param_names_lst[-1] == 'bias':
            no_decay.add(pn)
        elif 'norm' in pn:
            no_decay.add(pn)
        elif 'embedding' in param_names_lst or 'embed' in pn or 'pos_emb' in pn:
            no_decay.add(pn)
        elif re.search(layer_norm_pattern, pn) is not None:
            no_decay.add(pn)
        else:
            assert param_names_lst[-1] in ['weight', 'freq'] or re.search("w_[a-z]$", param_names_lst[-1].lower()) is not None
            decay.add(pn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set! keys: {param_dict.keys() - union_params}"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def extract_state_dict(state_dict, module_name):
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def remove_dir(path, should_ask=False):
    assert path.is_dir()
    if (not should_ask) or input(f"Remove directory : {path} ? [Y/n] ").lower() != 'n':
        shutil.rmtree(path)


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape[:2], f"{rewards.shape}, {ends.shape}, {values.shape}"  # (B, T, 1)
    if values.dim() == 3:
        assert values.shape[2] == 1
        values = rearrange(values, 'b t 1 -> b t')

    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = values[:, -1]
    for i in list(range(t - 1))[::-1]:
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        if len(kwargs) > 0:
            self.loss_total = sum(kwargs.values())
            self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}
        else:
            self.loss_total = None
            self.intermediate_losses = {}

    @classmethod
    def combine(cls, losses: list):
        combined = cls()
        combined.loss_total = sum([l.loss_total for l in losses if l.loss_total is not None])
        combined.intermediate_losses = {k: v for l in losses for k, v in l.intermediate_losses.items()}
        return combined

    def __truediv__(self, value):
        assert self.loss_total is not None
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class DictObs:
    def __init__(self, obs: dict[ObsModality, Tensor], **kwargs):
        assert len(set([o.shape[:2] for o in obs.values()])) == 1
        self.obs = obs

    def __getitem__(self, item):
        return DictObs({k: v[item] for k, v in self.obs.items()})

    def __setitem__(self, key, value):
        assert isinstance(value, DictObs)
        assert set(self.obs.keys()) == set(value.obs.keys()), f"keys mismatch"
        for k in self.obs.keys():
            self.obs[k] = value.obs[k]

    def __dict__(self):
        return self.obs

    @classmethod
    def apply(cls, obs, fn, **kwargs):
        pass



class RandomHeuristic(ABC):

    def __init__(self, action_space: gymnasium.spaces.Space):
        self.action_space = action_space

    def act(self, obs: dict[ObsModality, Tensor]) -> Tensor:
        batch_size = next(iter(obs.values())).shape[0]
        return torch.tensor(np.array([self.action_space.sample() for _ in range(batch_size)]))


class DiscreteRandomHeuristic(RandomHeuristic):
    def __init__(self, action_space: Union[gymnasium.spaces.Discrete, gymnasium.spaces.MultiDiscrete]):
        assert isinstance(action_space, (gymnasium.spaces.Discrete, gymnasium.spaces.MultiDiscrete)), f"Got {action_space}"
        super().__init__(action_space=action_space)

    def act(self, obs: dict[ObsModality, Tensor]) -> Tensor:
        return super().act(obs).long()


class ContinuousRandomHeuristic(RandomHeuristic):
    def __init__(self, action_space: gymnasium.spaces.Box):
        assert isinstance(action_space, gymnasium.spaces.Box)
        super().__init__(action_space)

    def act(self, obs: dict[ObsModality, Tensor]) -> Tensor:
        return super().act(obs).float()

def make_video(fname, fps, frames):
    assert frames.ndim == 4 # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3

    # Define the codec and quality parameters
    codec = 'libx264'
    quality = 5  # Quality scale (0-10, lower is better)

    # Create the video writer
    import imageio
    writer = imageio.get_writer(fname, fps=fps, codec=codec, quality=quality)

    for frame in frames:
        writer.append_data(frame)

    writer.close()

    print(f"Video saved as {fname}")


class VideoMaker:
    def __init__(self, file_name, fps: int = 15, codec: str = 'libx264', quality: int = 5):
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self.file_name = file_name

        self.writer = None

    def add_frame(self, frame: np.ndarray):
        if self.writer is None:
            self.writer = imageio.get_writer(self.file_name, fps=self.fps, codec=self.codec, quality=self.quality)
        self.writer.append_data(frame)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            print(f"Video saved as {self.file_name}")
            self.writer = None


class TrainerInfoHandler(ABC):
    @abstractmethod
    def signal_epoch_start(self):
        pass

    @abstractmethod
    def update_with_step_info(self, step_info: dict):
        pass

    @abstractmethod
    def get_epoch_info(self) -> dict:
        pass


class TokenizerInfoHandler(TrainerInfoHandler):
    def __init__(self, codebook_size: int):
        self.codebook_size = codebook_size
        self.global_token_counter = FreqDist()
        self.epoch_token_counter = FreqDist()
        self.codebook_norms = None

    def signal_epoch_start(self):
        self.epoch_token_counter = FreqDist()

    def update_with_step_info(self, step_info: dict):
        assert ObsModality.image.name in step_info
        # TODO: support all modalities if needed
        step_info = step_info[ObsModality.image.name]
        assert 'token_counts' in step_info
        self.epoch_token_counter += step_info['token_counts']
        self.global_token_counter += step_info['token_counts']
        if 'codebook_norms' in step_info:
            self.codebook_norms = step_info['codebook_norms']

    def get_epoch_info(self) -> dict:
        epoch_info = {
            'Global codebook usage': self.global_token_counter.B() / self.codebook_size,
            'Epoch codebook usage': self.epoch_token_counter.B() / self.codebook_size,
        }
        if self.codebook_norms is not None:
            epoch_info['Codebook norms'] = wandb.Histogram(self.codebook_norms.cpu().numpy())

        return epoch_info
class WorldmodelInfoHandler(TrainerInfoHandler):

    def __init__(self) -> None:
        super().__init__()
        self.buffer = defaultdict(list)
        # self.imagination_intrinsic_rewards = []

    def signal_epoch_start(self):
        self.buffer = defaultdict(list)

    def update_with_step_info(self, step_info: dict):
        for k, v in step_info.items():
            if k=="change_percents":
                for modality,mc in v.items():
                    self.buffer.setdefault(modality.name+"_change_percents", []).append(mc.mean())

    def get_epoch_info(self) -> dict:
        info = {}
        for k, v in self.buffer.items():
            if k.endswith('_change_percents'):
                info[k] = torch.stack(v).mean().item()

        return info


class ControllerInfoHandler(TrainerInfoHandler):

    def __init__(self) -> None:
        super().__init__()
        self.buffer = defaultdict(list)
        # self.imagination_intrinsic_rewards = []

    def signal_epoch_start(self):
        self.buffer = defaultdict(list)

    def update_with_step_info(self, step_info: dict):
        if 'imagined_rewards' in step_info:
            rewards = step_info['imagined_rewards']
        elif 'real_rewards' in step_info:
            rewards = step_info['real_rewards']
        for k, v in step_info.items():
            self.buffer[k].append(v)

    def get_epoch_info(self) -> dict:
        info = {}
        for k, v in self.buffer.items():
            if k == 'imagined_returns_scale' or k == 'real_returns_scale':
                info[k+'_mean'] = np.array(v).mean()
            else:
                info.update({
                    f'{k}_mean': torch.stack(v).mean().item(),
                    f'{k}_std': torch.stack(v).std().item(),
                    f'{k}_hist': wandb.Histogram(torch.stack(v).flatten().detach().cpu().numpy(), num_bins=10)
                })
        return info


class GradNormInfo:
    def __init__(self):
        self.epoch_grad_norms = []

    def reset(self):
        self.epoch_grad_norms.clear()

    def __call__(self, iter_grad_norm, *args, **kwargs):
        self.epoch_grad_norms.append(iter_grad_norm.detach().cpu().numpy())

    def get_info(self):
        return {
            'max_grad_norm': np.max(self.epoch_grad_norms),
            'mean_grad_norm': np.mean(self.epoch_grad_norms),
            'min_grad_norm': np.min(self.epoch_grad_norms)
        }


class RecurrentState:
    def __init__(self, state: Optional[torch.Tensor], n: Union[int, torch.LongTensor]):
        """
        When the recurrent state is a batch, n is a tensor.
        """
        self.state = state
        self.n = n

    def to_dict(self):
        return {'state': self.state, 'n': self.n}

    def cpu_clone(self):
        n = self.n if isinstance(self.n, int) else self.n.clone().detach().cpu()
        return RecurrentState(self.state.clone().detach().cpu(), n)

    def clone(self):
        state = self.state.clone() if self.state is not None else None
        n = self.n.clone() if isinstance(self.n, torch.Tensor) else self.n
        return RecurrentState(state, n)

    def to_device(self, device):
        if self.state is not None:
            self.state = self.state.to(device)
        if isinstance(self.n, torch.LongTensor):
            self.n = self.n.to(device)
        else:
            assert isinstance(self.n, int)

        return self


class NormalizedSoftmax(nn.Module):
    def __init__(self, softmax_dim: int, input_dim: int, one_hot_target_prob: float = 0.8, at_value: float = 3, *args, **kwargs):
        """
        Computes a new coefficient that multiplies the input to the softmax.
        The coefficient is determined such that for a d-dimensional one-hot vector,
        the hot entry when multiplied by `at_value` would have a probability of `one_hot_target_prob`
        """
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.one_hot_target_prob = one_hot_target_prob
        self.at_value = at_value
        self.coefficient = self._compute_coefficient()
        self.softmax = nn.Softmax(dim=softmax_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.softmax(self.coefficient * x)

    def _compute_coefficient(self):
        a = self.one_hot_target_prob
        d = self.input_dim
        return np.log((a * (d-1)) / (1-a)) / self.at_value


class RegressionHead(nn.Module):

    def __init__(
            self, in_features: int, sym_log_normalize: bool = False,
            device=None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sym_log_normalize = sym_log_normalize
        self.linear = nn.Linear(in_features, 1, device=device)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x).squeeze(-1)
        return sym_exp(out) if self.sym_log_normalize else out

    def compute_loss(self, x: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
        pred = self.linear(x)
        assert pred.shape == target.shape, f"{pred.shape} != {target.shape}"
        if self.sym_log_normalize:
            target = sym_log(target)

        loss = F.mse_loss(pred, target, reduction=reduction)
        return loss


class CategoricalRegressionHead(nn.Module):

    def __init__(
            self, in_features: int, out_features: int, support: Tensor = None, sym_log_normalize: bool = False,
            device=None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sym_log_normalize = sym_log_normalize
        self.linear = nn.Linear(in_features, out_features, device=device)

        if support is None:
            support = torch.linspace(-15, 15, out_features, device=device)
            if not sym_log_normalize:
                support = sym_exp(support)
            bias = -support.abs() / (support.max() / torch.log(support.abs().max()))
            self.linear.bias.data = bias
        else:
            assert support.numel() == out_features, f"{support.numel()}, {out_features}"
            bias = -support.abs()
            self.linear.bias.data = bias
        self.support = support

    def forward(self, x: Tensor) -> Tensor:
        out = self._compute_output(x)
        if self.sym_log_normalize:
            out = sym_exp(out)

        return out

    @torch.compile()
    def _compute_output(self, x: Tensor) -> Tensor:
        logits = self.linear(x)
        out = einsum(torch.softmax(logits, -1), self.support, '... k, k -> ...')
        return out

    def compute_loss(self, x: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
        pred = self._compute_output(x)
        if self.sym_log_normalize:
            target = sym_log(target)
            # pred = sym_log(pred)

        loss = F.mse_loss(pred, target, reduction=reduction)
        return loss

    def compute_loss_from_pred(self, pred: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
        if self.sym_log_normalize:
            target = sym_log(target)
            pred = sym_log(pred)

        loss = F.mse_loss(pred, target, reduction=reduction)
        return loss


class HLGaussCategoricalRegressionHead(nn.Module):

    def __init__(
            self, in_features: int, out_features: int, support: Tensor = None, sym_log_normalize: bool = False,
            device=None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sym_log_normalize = sym_log_normalize
        self.linear = nn.Linear(in_features, out_features, device=device)

        if support is None:
            support = sym_exp(torch.linspace(-15, 15, out_features, device=device))
        else:
            assert support.numel() == out_features, f"{support.numel()}, {out_features}"

        if sym_log_normalize:
            support = sym_log(support)
            self.linear.bias.data = -0.2 * support.clone().abs()
        else:
            bias = -support.abs() / (support.max() / torch.log(support.abs().max()))
            self.linear.bias.data = bias

        self.support = support
        self.hl_gauss_loss = HLGaussLoss(
            min_value=self.support[0],
            max_value=self.support[-1],
            num_bins=out_features,
            smoothness=0.75,
            device=device
        )

    def forward(self, x: Tensor) -> Tensor:
        logits = self.linear(x)
        out = self.hl_gauss_loss.transform_from_logits(logits)
        if self.sym_log_normalize:
            out = sym_exp(out)

        return out

    def compute_loss(self, x: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
        logits = self.linear(x)
        if self.sym_log_normalize:
            target = sym_log(target)
            # pred = sym_log(pred)

        # loss = F.mse_loss(pred, target, reduction=reduction)
        if logits.dim() > 2:
            logits = logits.flatten(0, -2)
        loss = self.hl_gauss_loss(logits, target.flatten(), reduction=reduction)
        return loss


class HLGaussLoss(nn.Module):
    # Taken from https://arxiv.org/pdf/2403.03950#page=22.10
    # Modified
    def __init__(self, min_value: float, max_value: float, num_bins: int, smoothness: float = 0.75, device=None):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.smoothness = smoothness
        self.support = torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32, device=device)
        self.bin_width = self.support[1] - self.support[0]

    def forward(self, logits: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        targets_probs = self.transform_to_probs(target)
        return F.cross_entropy(logits, targets_probs, reduction=reduction)

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        sigma = self._compute_sigma(target)
        cdf_evals = torch.special.erf(
            (self.support - target.unsqueeze(-1))
            / (torch.sqrt(torch.tensor(2.0)) * sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return einsum(probs, centers, '... k, k -> ...')

    def transform_from_logits(self, logits: Tensor) -> Tensor:
        return self.transform_from_probs(torch.softmax(logits, dim=-1))

    def _compute_sigma(self, target: torch.Tensor) -> torch.Tensor:
        return self.smoothness * self.bin_width


class SymLogHLGaussLoss(HLGaussLoss):

    def __init__(self, max_value: float, num_bins: int, smoothness: float, device=None):
        assert num_bins % 2 == 0, f"num_bins ({num_bins}) must be an even number."
        super().__init__(-max_value, max_value, num_bins, smoothness, device=device)
        xs = torch.linspace(-max_value, max_value, num_bins + 1, dtype=torch.float32, device=device)
        self.support = sym_exp(xs)
        logger.info(f"SymLogHLGaussLoss supports {self.support}")

    def _compute_sigma(self, target: torch.Tensor) -> torch.Tensor:
        target = torch.clamp(target, self.support[0], self.support[-1])
        target_bin_left_idx = torch.searchsorted(self.support, target, right=False)
        bin_width = self.support[target_bin_left_idx + 1] - self.support[target_bin_left_idx]
        return self.smoothness * bin_width.unsqueeze(-1)


class VectorQuantizer:
    def __init__(self, normalize: bool = False, device=None):
        # in DMC, angles are represented in radians, i.e., [-pi/2, pi/2]. this range is quantized uniformly (dense).
        self.normalize = normalize
        n0 = 4
        n1 = 32
        n2 = 32
        # self.positive_bins = np.sort(np.concatenate([
        #     np.exp(np.linspace(-10, np.log(np.pi / (2*n1)), n0))[:-1],
        #     np.linspace(0, np.pi / 2, n1),
        #     np.exp(np.linspace(np.log(np.pi / 2), 7, n2))[1:]
        # ]))
        self.positive_bins = np.sort(np.concatenate([
            np.linspace(0, np.log1p(np.pi), n1),
            np.linspace(np.log1p(np.pi), 6, n2)[1:]
        ]))
        self.support = np.sort(np.concatenate([-self.positive_bins[1:], self.positive_bins]))
        self.mid_pts = (self.support[:-1] + self.support[1:])/2
        self.support_pt = torch.from_numpy(self.support).float().to(device)
        self.mid_pts_pt = ((self.support_pt[:-1] + self.support_pt[1:]) / 2).contiguous()

    def to(self, device):
        self.support_pt = self.support_pt.to(device)
        self.mid_pts_pt = self.mid_pts_pt.to(device)

    @property
    def vocab_size(self):
        return self.support.size

    def vector_to_tokens(self, x: np.ndarray) -> np.ndarray:
        token_obs = np.searchsorted(self.mid_pts, x, side='right')
        return token_obs

    def vector_to_tokens_pt(self, x: Tensor) -> Tensor:
        token_obs = torch.searchsorted(self.mid_pts_pt, x.contiguous(), side='right')
        return token_obs

    def quantize_vector(self, x: np.ndarray) -> np.ndarray:
        obs_tokens = self.vector_to_tokens(x)
        return self.tokens_to_vector(obs_tokens)

    def tokens_to_vector(self, tokens: np.ndarray) -> np.ndarray:
        if self.normalize:
            n = self.support.size
            return 2*(tokens.astype(np.float32) / n) - 1
        return self.support[tokens]

    def tokens_to_vector_pt(self, tokens: Tensor) -> Tensor:
        self.support_pt = self.support_pt.to(tokens.device)

        if self.normalize:
            n = self.support.size
            return 2*(tokens.float() / n) - 1

        return self.support_pt[tokens]


class UniformVQ(VectorQuantizer):
    def __init__(self, vmin: float = -7, vmax: float = 7, support_size: int = 129, normalize: bool = False, device=None):
        super().__init__(normalize=normalize, device=device)
        self.vmin = vmin
        self.vmax = vmax
        self.support = np.linspace(vmin, vmax, support_size)
        self.mid_pts = (self.support[:-1] + self.support[1:]) / 2
        self.support_pt = torch.from_numpy(self.support).float()
        if device is not None:
            self.support_pt = self.support_pt.to(device)
        self.mid_pts_pt = (self.support_pt[:-1] + self.support_pt[1:]) / 2


class ByteToken2FP16Mapper:
    @property
    def vocab_size(self):
        # a1 = np.frombuffer(np.arange(31743, dtype=np.uint16).tobytes(), dtype=np.float16)
        # a2 = np.frombuffer(np.arange(32768, 64512, dtype=np.uint16).tobytes(), dtype=np.float16)
        # a = np.concatenate([a2, a1])
        s = 31743 + (64512 - 32769)
        return s

    def tokens_to_vector_pt(self, tokens: Tensor) -> Tensor:
        assert torch.all(tokens <= self.vocab_size-1)
        # shape = (*tokens.shape[:-1], tokens.shape[-1])
        shape = tokens.shape
        tokens_np = tokens.detach().cpu().numpy().astype(np.uint16)
        tokens_np[tokens_np > 31743] += (32769 - 31744)
        vector_np = np.frombuffer(tokens_np.tobytes(), dtype=np.float16).copy()
        vector = torch.from_numpy(vector_np).float().reshape(shape).to(tokens.device)
        return vector

    def vector_to_tokens(self, x: np.ndarray) -> np.ndarray:
        tokens = x.astype(np.float16).tobytes()
        tokens = np.frombuffer(tokens, dtype=np.uint16).copy()
        tokens[tokens >= 32769] -= (32769 - 31744)  # avoid NaNs
        tokens[tokens == 32768] = 0  # get rid of -0.
        return tokens

    def vector_to_tokens_pt(self, x: Tensor) -> Tensor:
        x_np = x.detach().cpu().numpy()
        vector_np = self.vector_to_tokens(x_np)
        return torch.from_numpy(vector_np).long().reshape(x.shape).to(x.device)


class LSTMCellWrapper(nn.Module):
    def __init__(self, lstm: nn.LSTM):
        super().__init__()
        assert lstm.bidirectional == False and lstm.num_layers == 1
        assert lstm.batch_first
        self.lstm = lstm

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor] = None):
        # LSTM Cell compatible forward. for sequences use the `forward_sequence` method
        assert x.dim() == 2  # (batch, features)
        x = x.unsqueeze(1)

        _, state = self.forward_sequence(x, state)

        return state

    def forward_sequence(self, x: Tensor, state: tuple[Tensor, Tensor] = None):
        assert x.dim() == 3

        if state is not None:
            h, c = state
            assert h.dim() == 2 and c.dim() == 2, f"h: {h.shape}; c: {c.shape}"
            state = (h.unsqueeze(0), c.unsqueeze(0))

        outputs, (h, c) = self.lstm(x, state)
        state = h.squeeze(0), c.squeeze(0)

        return outputs, state

