"""Sampler interface"""
import random
from abc import ABCMeta
from abc import abstractmethod

import numpy as np


class Sampler(metaclass=ABCMeta):
    """The base class for sampler currently enforces that all samplers have
    modes for drawing training and validation samples to train a model
    """

    BASE_MODES = ("train", "validate")

    def __init__(self,
                 random_seed=436):
        self.modes = list(self.BASE_MODES)
        self.mode = None
        np.random.seed(random_seed)
        random.seed(random_seed + 1)

    def set_mode(self, mode):
        if mode not in self.modes:
            raise ValueError(
                "Tried to set mode to be '{0}' but the only valid modes are "
                "{1}".format(mode, self.modes))
        self.mode = mode

    @abstractmethod
    def sample(self, batch_size=1):
        raise NotImplementedError

    @abstractmethod
    def get_data_and_targets(self, mode, batch_size, n_samples):
        raise NotImplementedError

    @abstractmethod
    def get_validation_set(self, batch_size, n_samples=None):
        raise NotImplementedError
