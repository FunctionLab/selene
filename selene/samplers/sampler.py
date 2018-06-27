"""This module provides the `Sampler` base class, which defines the
interface for sampling classes. These sampling classes should provide
a way to query some training/validation/test data for examples.

"""
import random
from abc import ABCMeta
from abc import abstractmethod

import numpy as np


class Sampler(metaclass=ABCMeta):
    """
    The base class for sampler currently enforces that all samplers
    have modes for drawing training and validation samples to train a
    model.

    Attributes
    ----------
    modes : list(str)
        A list of the names of the modes that the object may operate in.
    mode : str or None
        Default is `None`. The current mode that the object is operating in.

    Parameters
    ----------
    seed : int
        The value used to seed the random number generator.

    """

    BASE_MODES = ("train", "validate")
    """
    The types of modes that the `Sampler` object can run in.
    """

    def __init__(self, seed=436):
        """
        Constructs a new `Sampler` object.
        """
        self.modes = list(self.BASE_MODES)
        self.mode = None
        self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed + 1)

    def set_mode(self, mode):
        """
        Sets the sampling mode.

        Parameters
        ----------
        mode : str
            The name of the mode to use. It must be one of
            `Sampler.BASE_MODES`.

        Raises
        ------
        ValueError
            If `mode` is not a valid mode.

        """
        if mode not in self.modes:
            raise ValueError(
                "Tried to set mode to be '{0}' but the only valid modes are "
                "{1}".format(mode, self.modes))
        self.mode = mode

    @abstractmethod
    def sample(self, batch_size=1):
        """
        Fetches a mini-batch of the data from the sampler.

        Parameters
        ----------
        batch_size : int
            The size of the batch to retrieve.

        """
        raise NotImplementedError()

    @abstractmethod
    def get_data_and_targets(self, mode, batch_size, n_samples):
        """
        This method fetches a subset of the data from the sampler,
        divided into batches. This method also allows the user to
        specify what operating mode to run the sampler in when fetching
        the data.

        Parameters
        ----------
        mode : str
            The operating mode that the object should run in.
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int
            The total number of samples to retrieve.

        """
        raise NotImplementedError()

    @abstractmethod
    def get_validation_set(self, batch_size, n_samples=None):
        """
        This method returns a subset of validation data from the
        sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of validation examples
            to retrieve. If `None`, it will retrieve all validation
            data.

        """
        raise NotImplementedError()
