from typing import Union

import numpy as np
import torch
from torch import Tensor


def base_n_to_base_10(x, n, num_digits):
    c = n ** torch.arange(num_digits - 1, -1, -1, device=x.device)
    tokens = torch.matmul(x.float(), c.float()).long()
    return tokens


@torch.no_grad()
def base_10_to_base_n(x, n, num_digits):
    c = n ** torch.arange(num_digits - 1, -1, -1, device=x.device).long()

    r = x
    digits = []
    for i in range(num_digits):
        digits.append(r // c[i])
        r = r % c[i]

    return torch.stack(digits, dim=-1).long()


@torch.compile()
def atanh(x):
    return 0.5 * (x.log1p() - (-x).log1p())


# @torch.compile()
def sym_log(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def sym_log_np(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.sign(x) * np.log1p(np.abs(x))


# @torch.compile()  # causes the support tensor in HLGaussCategoricalRegressionHead to init on cuda:0 for some reason
def sym_exp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
