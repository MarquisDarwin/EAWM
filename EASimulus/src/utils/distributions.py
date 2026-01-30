from abc import ABC, abstractmethod

import numpy as np
import torch
from einops import einsum
from torch import Tensor
from torch.distributions import Normal, Categorical
from torch.distributions.utils import lazy_property


class Distribution(ABC):
    @abstractmethod
    def detached_copy(self):
        pass


class CategoricalDistribution(torch.distributions.Categorical, Distribution):

    def detached_copy(self):
        return CategoricalDistribution(logits=self.logits.detach().clone())


class MultiCategoricalDistribution(Distribution):
    def __init__(self, logits: Tensor, nvec: tuple[int, ...]):
        self.logits = logits
        if isinstance(nvec, np.ndarray):
            nvec = tuple(nvec)
        self.nvec = nvec

    def sample(self, sample_shape=torch.Size()):
        logits = torch.split(self.logits, self.nvec, dim=-1)
        return torch.stack([Categorical(logits=logits_i).sample() for logits_i in logits], dim=-1)

    def log_prob(self, value: Tensor) -> Tensor:
        assert value.shape[-1] == len(self.nvec), f"Got {value.shape} which is incompatible with nvec={self.nvec}"
        logits = torch.split(self.logits, self.nvec, dim=-1)
        log_probs = torch.stack([Categorical(logits=logits[i]).log_prob(value[..., i])
                                 for i in range(len(self.nvec))], dim=-1).sum(dim=-1)
        return log_probs

    def entropy(self) -> Tensor:
        logits = torch.split(self.logits, self.nvec, dim=-1)
        entropy = torch.stack([Categorical(logits=logits_i).entropy() for logits_i in logits], dim=-1).sum(dim=-1)
        return entropy

    def detached_copy(self):
        return MultiCategoricalDistribution(logits=self.logits.detach().clone(), nvec=self.nvec)


class DiagNormalDistribution(torch.distributions.Normal, Distribution):

    def log_prob(self, value):
        return torch.sum(super().log_prob(value), dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape)

    def entropy(self):
        return super().entropy().sum(dim=-1)

    def detached_copy(self):
        return DiagNormalDistribution(loc=self.loc.detach().clone(), scale=self.scale.detach().clone())

    def kl(self, other):
        return torch.distributions.kl.kl_divergence(
            Normal(self.loc, self.scale),
            Normal(other.loc, other.scale)
        ).sum(-1)


class SquashedDiagNormalDistribution(DiagNormalDistribution):

    def sample(self, sample_shape=torch.Size()):
        return torch.tanh(super().sample(sample_shape))

    def log_prob(self, value):
        # Following the SAC paper (see https://arxiv.org/pdf/1801.01290#page=12.25)
        # avoid NaNs by clamping to [-1+eps, 1-eps]
        assert torch.all(value >= -1) and torch.all(value <= 1)
        eps = 1e-6
        normal_value = torch.atanh(value.clamp(-1 + eps, 1 - eps))
        # normal_value = atanh(value.clamp(-1 + eps, 1 - eps))

        normal_log_probs = super().log_prob(normal_value)

        corrected_log_probs = normal_log_probs - torch.sum(torch.log(1 - value ** 2 + eps), dim=-1)

        return corrected_log_probs

    def entropy(self):
        # TODO: FIX THIS! after tanh the entropy is not the same!
        return super().entropy()

    def detached_copy(self):
        return SquashedDiagNormalDistribution(loc=self.loc.detach().clone(), scale=self.scale.detach().clone())


class QuantizedContinuousDistribution3(Categorical):

    def __init__(self, interval: tuple[float, float] = (-1, 1), probs=None, logits=None, validate_args=None):
        """
        Assume last two dims are (num dims, num categories)
        """
        self.interval = interval
        super().__init__(probs, logits, validate_args)

        self.quantized_interval = torch.linspace(
            start=interval[0],
            end=interval[1],
            steps=self.logits.shape[-1],
            device=self.logits.device,
            dtype=torch.float32
        )
        self._mid_pts = (self.quantized_interval[1:] + self.quantized_interval[:-1]) / 2

    def sample(self, sample_shape=torch.Size()):
        samples = super().sample(sample_shape)
        return self.quantized_interval[samples]

    def log_prob(self, value):
        samples = torch.searchsorted(self._mid_pts, value)
        return super().log_prob(samples).sum(dim=-1)

    def entropy(self):
        return super().entropy().sum(dim=-1)

    def kl(self, other):
        p = self.probs
        return (p * (torch.log(p) - torch.log(other.probs))).sum(-1).sum(-1)

    def detached_copy(self):
        return QuantizedContinuousDistribution(interval=self.interval, logits=self.logits.detach().clone())


class QuantizedInterval(ABC):
    @property
    @abstractmethod
    def quantization_values(self) -> Tensor:
        pass

    @property
    @abstractmethod
    def sigmas(self) -> Tensor:
        pass


class UniformQI(QuantizedInterval):
    def __init__(self, num_values: int, interval_endpoints: tuple[float, float] = (-1, 1),
                 smoothness: float = 2.0, device='cuda'):
        self.smoothness = smoothness
        self.quantized_interval = torch.linspace(
            start=interval_endpoints[0],
            end=interval_endpoints[1],
            steps=num_values,
            device=device,
            dtype=torch.float32
        )
        bin_width = self.quantized_interval[1] - self.quantized_interval[0]
        self._sigmas = torch.ones_like(self.quantized_interval) * bin_width * smoothness

    @property
    def quantization_values(self) -> Tensor:
        return self.quantized_interval

    @property
    def sigmas(self) -> Tensor:
        return self._sigmas


class CustomQI(QuantizedInterval):
    def __init__(self, interval_quantization_values: Tensor, sigmas=None, base_sigma_choice: str = 'min',
                 smoothness: float = 2.0, device='cuda'):
        super().__init__()
        self.device = device
        self.smoothness = smoothness
        self.base_sigma_choice = base_sigma_choice
        self.interval_quantization_values = interval_quantization_values.to(device)

        if sigmas is None:
            self._sigmas = torch.ones_like(self.interval_quantization_values)
            margins = self.quantization_values[1:] - self.quantization_values[:-1]
            self._sigmas[0] = margins[0]
            self._sigmas[-1] = margins[-1]
            if self.base_sigma_choice == 'min':
                self._sigmas[1:-1] = torch.where(margins[:-1] < margins[1:], margins[:-1], margins[1:])
            elif self.base_sigma_choice == 'max':
                self._sigmas[1:-1] = torch.where(margins[:-1] < margins[1:], margins[1:], margins[:-1])
            else:
                raise ValueError(f'Unknown base sigma choice: {self.base_sigma_choice}')

            self._sigmas = self._sigmas * self.smoothness
        else:
            self._sigmas = sigmas

    @property
    def quantization_values(self) -> Tensor:
        return self.interval_quantization_values

    @property
    def sigmas(self) -> Tensor:
        return self._sigmas


class QuantizedContinuousDistribution2(Categorical, Distribution):

    def __init__(self, interval: QuantizedInterval = None, probs=None, logits=None, validate_args=None, normalize_log_prob: bool = False):
        """
        Assume last two dims are (num dims, num categories)
        The interval is quantized to N points, where N is the number of categories.
        Each category corresponds to a certain quantization value v_k, and is associated with a Gaussian distribution
        centered around v_k with std sigma (hyperparameter, ideally should be at least bin_width).
        Actions are sampled as follows:
        a_t ~ p_k where p_k = N(v_k, sigma) is the pdf at a_t of a Normal dist. centered at v_k,
        k ~ D = softmax(y), y = logits.
        Then, the probability of the action is given by
        p(a_t) = sum_{k=1}^{N} D_k * p_k(a_t).
        """
        super().__init__(probs, logits, validate_args)
        self.normalize_log_prob = normalize_log_prob

        if interval is None:
            interval = UniformQI(num_values=self.logits.shape[-1], smoothness=0.5, device=self.logits.device)

        assert interval.quantization_values.numel() == self.logits.shape[-1], f"{interval.quantization_values.numel()}; {self.logits.shape}"
        self.interval = interval

        self.normal_dist = Normal(self.interval.quantization_values, self.interval.sigmas)
        self.support_densities = torch.exp(self.normal_dist.log_prob(self.normal_dist.loc.unsqueeze(-1)))
        self.support_size = interval.quantization_values.numel()

    def _smoothed_probs(self):
        weighted_densities = einsum(self.probs, self.support_densities, '... d k, k2 k -> ... d k2')
        return weighted_densities / weighted_densities.sum(dim=-1, keepdim=True)

    @lazy_property
    def smoothed_probs(self):
        return self._smoothed_probs()

    def sample(self, sample_shape=torch.Size()):
        return self.sample_mode_smoothed(sample_shape)

    def sample_mode(self, sample_shape=torch.Size()):
        k_samples = super().sample(sample_shape)
        normal_samples = self.normal_dist.mean[k_samples]
        return normal_samples

    def sample_mode_smoothed(self, sample_shape=torch.Size()):
        smoothed_probs = self.smoothed_probs
        modes_samples = Categorical(probs=smoothed_probs).sample()
        normal_samples = self.normal_dist.mean[modes_samples]
        return normal_samples

    def sample_mode_and_value(self, sample_shape=torch.Size()):
        k_samples = super().sample(sample_shape)
        normal_samples = self.normal_dist.sample(sample_shape)
        normal_samples = normal_samples[k_samples]
        return normal_samples

    def log_prob(self, value):
        return self._log_prob(value).mean(dim=-1)  # In our case, this is not log probability, but log pdf!

    def _log_prob(self, value):
        assert value.shape == self.logits.shape[:-1], f"{value.shape}, {self.logits.shape}"
        centers = (self.normal_dist.loc[:-1] + self.normal_dist.loc[1:]) / 2
        indices = torch.searchsorted(centers, value)  # (b, t, a)
        # select the appropriate index along the last axis:
        p = self.smoothed_probs.view(-1, self.support_size)[torch.arange(indices.numel()), indices.view(-1)]
        p = p.view(indices.size())
        log_p = torch.log(p)
        return log_p

    def mean(self):
        return einsum(self.smoothed_probs, self.normal_dist.mean, '... k, k -> ...')

    def entropy(self):
        return super().entropy().mean(dim=-1)

    def kl(self, other):
        assert not self.normalize_log_prob
        assert torch.all(self.normal_dist.loc == other.normal_dist.loc)
        assert torch.all(self.normal_dist.scale == other.normal_dist.scale)
        normal_pdfs = self.normal_dist.log_prob(self.normal_dist.loc.unsqueeze(-1))
        log_prob = torch.log((self.probs.unsqueeze(-2) * torch.exp(normal_pdfs)).sum(dim=-1))
        other_log_prob = torch.log((other.probs.unsqueeze(-2) * torch.exp(normal_pdfs)).sum(dim=-1))
        kl = (torch.exp(log_prob) * (log_prob - other_log_prob)).sum(dim=-1)
        # return self.probs * (self.logits - other.logits)  # Categorical logits are normalized
        return kl.mean(dim=-1)

    def detached_copy(self):
        return QuantizedContinuousDistribution2(self.interval, logits=self.logits.detach().clone(), normalize_log_prob=self.normalize_log_prob)


class QuantizedContinuousDistribution(Categorical):

    def __init__(self, probs=None, logits=None, validate_args=None, interval: QuantizedInterval = None):
        """
        Simple categorical distributions + quantization of continuous intervals for continuous action spaces.
        """
        super().__init__(probs, logits, validate_args)

        if interval is None:
            interval = UniformQI(num_values=self.logits.shape[-1], smoothness=0.5, device=self.logits.device)

        assert interval.quantization_values.numel() == self.logits.shape[-1], f"{interval.quantization_values.numel()}; {self.logits.shape}"
        self.interval = interval

    def sample(self, sample_shape=torch.Size()):
        return self.sample_mode(sample_shape)

    def sample_mode(self, sample_shape=torch.Size()):
        k_samples = super().sample(sample_shape)
        quntized_samples = self.interval.quantization_values[k_samples]
        return quntized_samples

    def log_prob(self, value):
        return self._log_prob(value).mean(dim=-1)  # In our case, this is not log probability, but log pdf!

    def _log_prob(self, value):
        assert value.shape == self.logits.shape[:-1], f"{value.shape}, {self.logits.shape}"
        quantized_values = self.interval.quantization_values
        centers = (quantized_values[:-1] + quantized_values[1:]) / 2
        indices = torch.searchsorted(centers, value)  # (b, t, a)
        # select the appropriate index along the last axis:
        p = self.probs.view(-1, quantized_values.numel())[torch.arange(indices.numel()), indices.view(-1)]
        p = p.view(indices.size())
        log_p = torch.log(p)
        return log_p

    def mean(self):
        return einsum(self.probs, self.interval.quantization_values, '... k, k -> ...')

    def entropy(self):
        return super().entropy().mean(dim=-1)


def sample_categorical(logits: Tensor, temp: float = 1.0) -> Tensor:
    # assert logits.dim() == 3  # (batch, seq len, k-options to sample)
    return Categorical(logits=logits / temp).sample().long()


class Sampler(ABC):
    @abstractmethod
    def __call__(self, logits: Tensor, *args, **kwargs) -> Tensor:
        pass


class CategoricalSampler(Sampler):
    def __init__(self, temp: float = 1.0):
        self.temp = temp

    def __call__(self, logits: Tensor, *args, **kwargs) -> Tensor:
        return sample_categorical(logits, temp=self.temp)


class MultiCategoricalSampler(Sampler):
    def __init__(self, vocab_sizes: np.ndarray, temp: float = 1.0):
        assert vocab_sizes.ndim == 1
        self.vocab_sizes = vocab_sizes
        self.temp = temp

    def __call__(self, logits: Tensor, *args, **kwargs) -> Tensor:
        assert logits.shape[-1] == self.vocab_sizes.sum()
        separate_logits = torch.split(logits, self.vocab_sizes.tolist(), dim=-1)
        tokens = torch.stack([sample_categorical(logits_i, temp=self.temp) for logits_i in separate_logits], dim=-1)
        return tokens
