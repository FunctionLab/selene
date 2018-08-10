"""
This module provides the `MultiFileSampler` class, which uses a
FileSampler for each mode of sampling (train, test, validation).
The MultiFileSampler is therefore a subclass of Sampler.
"""

from .sampler import Sampler


class MultiFileSampler(Sampler):
    """
    This sampler contains individual file samplers for each mode.
    The file samplers parse .bed/.mat files that correspond to
    training, validation, and testing and MultiFileSampler calls on
    the correct file sampler to draw samples for a given mode.

    Attributes
    ----------
    train_sampler : selene_sdk.samplers.file_samplers.FileSampler
        Load your training data as a `FileSampler` before passing it
        into the `MultiFileSampler` constructor.
    validate_sampler : selene_sdk.samplers.file_samplers.FileSampler
        The validation dataset file sampler.
    features : list(str)
        The list of features the model should predict
    test_sampler : None or selene_sdk.samplers.file_samplers.FileSampler, optional
        Default is None. The test file sampler is optional.
    save_datasets : list(str), optional
        Default is None. Currently, we are only including these parameters
        so that `MultiFileSampler` is consistent with `Sampler`. The save
        dataset functionality for MultiFileSampler has not been defined
        yet.
    output_dir : str or None, optional
        Default is None. Used if the sampler has any data or logging
        statements to save to file. Currently not useful for
        `MultiFileSampler`.

    Attributes
    ----------
    modes : list(str)
        A list of the names of the modes that the object may operate in.
    mode : str or None
        Default is `None`. The current mode that the object is operating in.

    """
    def __init__(self,
                 train_sampler,
                 validate_sampler,
                 features,
                 test_sampler=None,
                 save_datasets=[],
                 output_dir=None):
        """
        Constructs a new `MultiFileSampler` object.
        """
        super(MultiFileSampler, self).__init__(
            features,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self._samplers = {
            "train": train_sampler,
            "validate": validate_sampler
        }

        self._index_to_feature = {
            i: f for (i, f) in enumerate(features)
        }

        if test_sampler is not None:
            self.modes.append("test")
            self._samplers["test"] = test_sampler

    def set_mode(self, mode):
        """
        Sets the sampling mode.

        Parameters
        ----------
        mode : str
            The name of the mode to use. It must be one of
            `Sampler.BASE_MODES` ("train", "validate") or "test" if
            the test data is supplied.

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
        return self._index_to_feature[index]

    def sample(self, batch_size=1):
        """
        Fetches a mini-batch of the data from the sampler.

        Parameters
        ----------
        batch_size : int, optional
            Default is 1. The size of the batch to retrieve.

        """
        return self._samplers[self.mode].sample(batch_size)

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
        return self._samplers[mode].get_data_and_targets(
            batch_size, n_samples)

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
        return self._samplers["validate"].get_data_and_targets(
            batch_size, n_samples)

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
        return self._samplers["test"].get_data_and_targets(
            batch_size, n_samples)

    def save_dataset_to_file(self, mode, close_filehandle=False):
        """
        We implement this function in this class only because the
        TrainModel class calls this method. In the future, we will
        likely remove this method or implement a different way
        of "saving the data" for file samplers. For example, we
        may only output the row numbers sampled so that users may
        reproduce exactly what order the data was sampled.

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
        return None
