from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from utils import CategoricalDistribution, SquashedDiagNormalDistribution, QuantizedContinuousDistribution, \
    MultiModalObs, Distribution, MultiCategoricalDistribution


class ValuesInfo(ABC):
    @property
    @abstractmethod
    def value_means(self):
        pass

    @abstractmethod
    def detached_copy(self):
        pass


class ContinuousValuesInfo(ValuesInfo):
    def __init__(self, values: Tensor) -> None:
        super().__init__()
        self.values = values

    @property
    def value_means(self):
        return self.values

    def detached_copy(self):
        return self.values.detach().clone()


class CategoricalValuesInfo(ValuesInfo):
    def __init__(self, values_logits: Tensor, values: Tensor) -> None:
        super().__init__()
        self.values_logits = values_logits
        self.values = values

    @property
    def value_means(self):
        return self.values

    def detached_copy(self):
        return self.values.detach().clone()


@dataclass
class ActorOutput:

    @abstractmethod
    def get_actions_distributions(self, temperature: float = 1.0) -> torch.distributions.Distribution:
        pass


@dataclass
class CriticOutput:

    @property
    @abstractmethod
    def means_values(self) -> Tensor:
        pass

    @abstractmethod
    def get_value_info(self) -> ValuesInfo:
        pass


@dataclass
class ContinuousCriticOutput(CriticOutput, ABC):
    value_means: Tensor

    @property
    def means_values(self) -> Tensor:
        return self.value_means

    def get_value_info(self):
        return ContinuousValuesInfo(self.means_values)


@dataclass
class CategoricalCriticOutput(CriticOutput):
    value_logits: Tensor
    values: Tensor

    @property
    def means_values(self) -> Tensor:
        return self.values

    def get_value_info(self):
        return CategoricalValuesInfo(self.value_logits, self.values)


@dataclass
class DiscreteActorOutput(ActorOutput):
    logits_actions: torch.FloatTensor

    def get_actions_distributions(self, temperature: float = 1.0) -> CategoricalDistribution:
        return CategoricalDistribution(logits=self.logits_actions / temperature)


@dataclass
class MultiDiscreteActorOutput(ActorOutput):
    logits_actions: torch.FloatTensor
    nvec: tuple[int, ...]

    def get_actions_distributions(self, temperature: float = 1.0) -> MultiCategoricalDistribution:
        return MultiCategoricalDistribution(logits=self.logits_actions / temperature, nvec=self.nvec)


@dataclass
class ContinuousActorOutput(ActorOutput):
    actions_means: torch.FloatTensor
    actions_stds: torch.FloatTensor

    def get_actions_distributions(self, temperature: float = 1.0) -> SquashedDiagNormalDistribution:
        return SquashedDiagNormalDistribution(loc=self.actions_means, scale=self.actions_stds * temperature)


@dataclass
class QuantizedContinuousActorOutput(ActorOutput):
    logits_actions: Tensor

    def get_actions_distributions(self, temperature: float = 1.0) -> QuantizedContinuousDistribution:
        return QuantizedContinuousDistribution(logits=self.logits_actions / temperature)


@dataclass
class ImagineOutput:
    observations: MultiModalObs
    actions: torch.LongTensor
    actions_distributions: Distribution
    values_info: ValuesInfo
    q_values_info: ValuesInfo
    rewards: torch.FloatTensor
    ends: torch.BoolTensor

    def detached_copy(self):
        return ImagineOutput(
            {k: v.detach().clone() for k, v in self.observations.items()},
            self.actions.detach().clone(),
            self.actions_distributions.detached_copy(),
            values_info=self.values_info.detached_copy(),
            q_values_info=None,  # self.q_values_info.detached_copy(),
            rewards=self.rewards.detach().clone(),
            ends=self.ends.detach().clone()
        )
