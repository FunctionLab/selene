"""
This module provides the `MultiSampler` class, which accepts
either an online sampler or a file sampler for each mode of
sampling (train, test, validation).
MultiSampler is a subclass of Sampler.
"""
import numpy as np
from torch.utils.data import DataLoader

from .sampler import Sampler
from .file_samplers import FileSampler


def MultiFileSampler(*args, **kwargs):
    """
    `MultiFileSampler` is deprecated and will be removed from future
    versions of Selene. Please use `MultiSampler` instead. This function
    maintains backward compatibility for code that uses `MultiFileSampler`,
    but we will remove this function in future. Please refer to the
    `MultiSampler` documentation for usage.
    """
    from warnings import warn
    warn("MultiFileSampler is deprecated and will be removed from future "
    "versions of Selene. Please use MultiSampler instead.")
    return MultiSampler(*args, **kwargs)


class MultiSampler(Sampler):
    """
    This sampler draws samples from individual file samplers or data loaders
    that corresponds to training, validation, and testing (optional) modes.
    MultiSampler calls on the correct file sampler or data loader to draw
    samples for a given mode. Example file samplers are under
    `selene_sdk.samplers.file_samplers` and example data loaders are under
    `selene_sdk.samplers.dataloaders`.

    MultiSampler can use either file samplers or data loaders for
    different modes. Using data loaders for some modes while using file samplers
    for other modes are also allowed. The file samplers parse data files
    (e.g. bed, mat, or hdf5). The data loaders provide multi-worker iterators
    that draw samples from online samplers (i.e. on-the-fly sampling). As data
    loaders support parallel sampling, they are generally recommended for
    sampling speed.

    Parameters
    ----------
    train_sampler : selene_sdk.samplers.file_samplers.FileSampler or \
                    selene_sdk.samplers.dataloader.DataLoader
        Load your training data as a `FileSampler` or `DataLoader`
    validate_sampler : FileSampler or DataLoader
        The validation dataset file sampler or data loader.
    features : list(str)
        The list of features the model should predict
    test_sampler : None or FileSampler or DataLoader, optional
        Default is None. The test file sampler is optional.
    mode : str, optional
        Default is "train". Must be one of `{train, validate, test}`. The
        starting mode in which to run the sampler.
    save_datasets : list(str) or None, optional
        Default is None. Currently, we are only including this parameter
        so that `MultiSampler` is consistent with the `Sampler` interface.
        The save dataset functionality for MultiSampler has not been
        defined yet.
    output_dir : str or None, optional
        Default is None. Only used if the sampler has any data or
        logging statements to save to file. Currently not used in
        `MultiSampler`.

    Attributes
    ----------
    modes : list(str)
        A list of the modes that the object may operate in.
    mode : str or None
        Default is `None`. The current mode that the object is operating in.

    """
    def __init__(self,
                 train_sampler,
                 validate_sampler,
                 features,
                 test_sampler=None,
                 mode="train",
                 save_datasets=[],
                 output_dir=None):
        """
        Constructs a new `MultiSampler` object.
        """
        super(MultiSampler, self).__init__(
            features,
            save_datasets=save_datasets,
            output_dir=output_dir)
        self._samplers = {
            "train": train_sampler if (isinstance(train_sampler, FileSampler) or
                                       isinstance(train_sampler, Sampler)) \
                     else None,
            "validate": validate_sampler if (isinstance(validate_sampler, FileSampler) or
                                             isinstance(validate_sampler, Sampler)) \
                        else None
        }

        self._dataloaders = {
            "train": train_sampler if isinstance(train_sampler, DataLoader) \
                     else None,
            "validate": validate_sampler if isinstance(validate_sampler, DataLoader) \
                        else None
        }

        self._iterators = {
            "train": iter(self._dataloaders["train"]) \
                if self._dataloaders["train"] else None,
            "validate": iter(self._dataloaders["validate"]) \
                if self._dataloaders["validate"] else None
        }

        self._index_to_feature = {i: f for (i, f) in enumerate(features)}

        if test_sampler is not None:
            self.modes.append("test")
            self._samplers["test"] = \
                test_sampler if (isinstance(test_sampler, FileSampler) or
                                 isinstance(test_sampler, Sampler)) else None
            self._dataloaders["test"] = \
                test_sampler if isinstance(test_sampler, DataLoader) else None
            self._iterators["test"] = iter(self._dataloaders["test"]) \
                if self._dataloaders["test"] else None

        self.mode = mode

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

    def _set_batch_size(self, batch_size, mode=None):
        """
        Sets the batch size for DataLoader for the specified mode,
        if the specified batch_size does not equal the current batch_size.
        Parameters
        ----------
        batch_size : int
            The batch size for the mode.
        mode : str, optional
            Default is None. The  mode to set batch_size
            If None, will use the current mode `self.mode`.
        """
        if mode is None:
            mode = self.mode

        if self._dataloaders[mode]:
            batch_size_matched = True
            if self._dataloaders[mode].batch_sampler:
                if self._dataloaders[mode].batch_sampler.batch_size != batch_size:
                    self._dataloaders[mode].batch_sampler.batch_size = batch_size
                    batch_size_matched = False
            else:
                if self._dataloaders[mode].batch_size != batch_size:
                    self._dataloaders[mode].batch_size = batch_size
                    batch_size_matched = False

            if not batch_size_matched:
                print("Reset data loader for mode {0} to use the new batch "
                      "size: {1}.".format(mode, batch_size))
                self._iterators[mode] = iter(self._dataloaders[mode])

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

    def sample(self, batch_size=1, mode=None):
        """
        Fetches a mini-batch of the data from the sampler.

        Parameters
        ----------
        batch_size : int, optional
            Default is 1. The size of the batch to retrieve.
        mode : str, optional
            Default is None. The operating mode that the object should run in.
            If None, will use the current mode `self.mode`.
        """
        mode = mode if mode else self.mode
        if self._samplers[mode]:
            return self._samplers[mode].sample(batch_size)
        else:
            self._set_batch_size(batch_size, mode=mode)
            try:
                data, targets = next(self._iterators[mode])
                return data.numpy(), targets.numpy()
                #data, targets, ind = next(self._iterators[mode])
                #return data.numpy(), targets.numpy(), ind.numpy()
            except StopIteration:
                #If DataLoader iterator reaches its length, reinitialize
                self._iterators[mode] = iter(self._dataloaders[mode])
                #data, targets, ind = next(self._iterators[mode])
                #return data.numpy(), targets.numpy(), ind.numpy()
                data, targets = next(self._iterators[mode])
                return data.numpy(), targets.numpy()

    def get_data_and_targets(self, batch_size, n_samples=None, mode=None):
        """
        This method fetches a subset of the data from the sampler,
        divided into batches. This method also allows the user to
        specify what operating mode to run the sampler in when fetching
        the data.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is None. The total number of samples to retrieve.
            If `n_samples` is None, if a FileSampler is specified for the
            mode, the number of samplers returned is defined by the FileSampler,
            or if a Dataloader is specified, will set `n_samples` to 32000
            if the mode is `validate`, or 640000 if the mode is `test`.
            If the mode is `train` you must have specified a value for
            `n_samples`.
        mode : str, optional
            Default is None. The operating mode that the object should run in.
            If None, will use the current mode `self.mode`.
        """
        mode = mode if mode else self.mode
        if self._samplers[mode]:
            return self._samplers[mode].get_data_and_targets(
                batch_size, n_samples)
        else:
            if n_samples is None:
                if mode == 'validate':
                    n_samples = 32000
                elif mode == 'test':
                    n_samples = 640000
            self._set_batch_size(batch_size, mode=mode)
            data_and_targets = []
            targets_mat = []
            #ind_mat = []
            count = batch_size
            while count < n_samples:
                #data, tgts, ind = self.sample(batch_size=batch_size, mode=mode)
                #data_and_targets.append((data, tgts, ind))
                output = self.sample(batch_size=batch_size, mode=mode)
                data_and_targets.append(output)
                targets_mat.append(output[1])
                #ind_mat.append(ind)
                count += batch_size
            remainder = batch_size - (count - n_samples)
            #data, tgts, ind = self.sample(batch_size=remainder)
            #data_and_targets.append((data, tgts, ind))
            #targets_mat.append(tgts)
            #ind_mat.append(ind)
            output = self.sample(batch_size=remainder)
            data_and_targets.append(output)
            targets_mat.append(output[1])
            targets_mat = np.vstack(targets_mat)
            #ind_mat = np.hstack(ind_mat)
            #return data_and_targets, targets_mat, ind_mat
            return data_and_targets, targets_mat

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
            retrieve. If `n_samples` is None,
            then if a FileSampler is specified for the 'validate' mode, the
            number of samplers returned is defined by the FileSample,
            or if a Dataloader is specified, will set `n_samples` to
            32000.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets` sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S = n_samples`.

        Raises
        ------
        ValueError
            If no test partition of the data was specified during
            sampler initialization.
        """
        return self.get_data_and_targets(
            batch_size, n_samples, mode="validate")

    def get_test_set(self, batch_size, n_samples=None):
        """
        This method returns a subset of testing data from the
        sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is None. The total number of test examples to
            retrieve. If `n_samples` is None,
            then if a FileSampler is specified for the 'test' mode, the
            number of samplers returned is defined by the FileSample,
            or if a Dataloader is specified, will set `n_samples` to
            640000.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets` sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S = n_samples`.

        Raises
        ------
        ValueError
            If no test partition of the data was specified during
            sampler initialization.
        """
        return self.get_data_and_targets(
            batch_size, n_samples, mode="test")

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
