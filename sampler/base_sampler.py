"""Sampler interface"""
import random

import numpy as np


class BaseSampler(object):
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

    def sample(self, batch_size=1):
        raise NotImplementedError

    def get_data_and_targets(self, mode, batch_size, n_samples):
        raise NotImplementedError

    def get_validation_set(self, batch_size, n_samples=None):
        raise NotImplementedError
