"""
This module provides the `FileSampler` base class, which defines the
interface for classes that draw samples from a particular file type.

We recognize that there is confusion regarding the different samplers
Selene implements and will resolve this issue soon.
"""
from abc import ABCMeta
from abc import abstractmethod


class FileSampler(metaclass=ABCMeta):
    """
    Classes that implement `FileSampler` can be initialized
    in any way, but must implement the methods `sample`,
    `get_data_and_targets`, and `get_data`.
    """

    """
    The types of modes that the `Sampler` object can run in.
    """

    def __init__(self):
        """
        Constructs a new `FileSampler` object.
        """

    @abstractmethod
    def sample(self, batch_size=1):
        """
        Fetches a mini-batch of the data from the sampler.

        Parameters
        ----------
        batch_size : int, optional
            Default is 1. The size of the batch to retrieve.

        """
        raise NotImplementedError()

    @abstractmethod
    def get_data_and_targets(self, batch_size, n_samples):
        """
        This method fetches a subset of the sequence data and
        corresponding targets from the sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int
            The total number of samples to retrieve.

        """
        raise NotImplementedError()

    @abstractmethod
    def get_data(self, batch_size, n_samples):
        """
        This method fetches a subset of the data from the sampler,
        divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int
            The total number of samples to retrieve.

        """
        raise NotImplementedError()
