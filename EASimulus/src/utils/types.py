from enum import Enum

from torch import Tensor


class ObsModality(Enum):
    image = 'image'
    vector = 'vector'
    token = 'token'
    token_2d = 'token_2d'


MultiModalObs = dict[ObsModality, Tensor]
