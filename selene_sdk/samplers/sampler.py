"""This module provides the `Sampler` base class, which defines the
interface for sampling classes. These sampling classes should provide
a way to query some training/validation/test data for examples.
"""
from abc import ABCMeta
from abc import abstractmethod
import os


class Sampler(metaclass=ABCMeta):
    """
    The base class for sampler currently enforces that all samplers
    have modes for drawing training and validation samples to train a
    model.

    Parameters
    ----------
    features : list(str)
        The list of features (classes) the model predicts.
    save_datasets : list(str), optional
        Default is `[]` the empty list. The list of modes for which we should
        save sampled data to file (1 or more of ['train', 'validate', 'test']).
    output_dir : str or None, optional
        Default is None. Path to the output directory. Used if we save
        any of the data sampled. If `save_datasets` is non-empty,
        `output_dir` must be a valid path. If the directory does not
        yet exist, it will be created for you.

    Attributes
    ----------
    modes : list(str)
        A list of the names of the modes that the object may operate in.
    mode : str or None
        The current mode that the object is operating in.

    """
    BASE_MODES = ("train", "validate")
    """
    The types of modes that the `Sampler` object can run in.
    """

    def __init__(self, features, save_datasets=[], output_dir=None):
        """
        Constructs a new `Sampler` object.
        """
        self.modes = list(self.BASE_MODES)
        self.mode = None

        self._features = features

        self._save_datasets = {}
        for mode in save_datasets:
            self._save_datasets[mode] = []

        self._output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

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
    def get_feature_from_index(self, index):
        """
        Returns the feature corresponding to an index in the feature
        vector.

        Parameters
        ----------
        index : int
            The index of the feature to retrieve the name for.

        Returns
        -------
        str
            The name of the feature occurring at the specified index.

        """
        raise NotImplementedError()

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
        n_samples : int, optional
            Default is None. The total number of validation examples to
            retrieve. Handling for `n_samples=None` should be done by
            all classes that subclass `selene_sdk.samplers.Sampler`.

        """
        raise NotImplementedError()

    @abstractmethod
    def get_test_set(self, batch_size, n_samples=None):
        """
        This method returns a subset of testing data from the
        sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of validation examples
            to retrieve. If `None`, 640000 examples are retrieved.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.

        Raises
        ------
        ValueError
            If no test partition of the data was specified during
            sampler initialization.

        """
        raise NotImplementedError()

    @abstractmethod
    def save_dataset_to_file(self, mode, close_filehandle=False):
        """
        Save samples for each partition (i.e. train/validate/test) to
        disk.

        Parameters
        ----------
        mode : str
            Must be one of the modes specified in `save_datasets` during
            sampler initialization.
        close_filehandle : bool, optional
            Default is False. `close_filehandle=True` assumes that all
            data corresponding to the input `mode` has been saved to
            file and `save_dataset_to_file` will not be called with
            `mode` again.

        """
        raise NotImplementedError()
