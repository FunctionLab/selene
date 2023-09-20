"""
This module provides the `SamplerDataLoader` and  `SamplerDataset` classes,
which allow parallel sampling for any Sampler using
torch DataLoader mechanism.
"""
import random
import  sys

import h5py
import numpy as np
import torch

from functools import wraps
from torch.utils.data import Dataset, DataLoader


class _SamplerDataset(Dataset):
    """
    This class provides a Dataset interface that wraps around a Sampler.
    `_SamplerDataset` is used internally by `SamplerDataLoader`.

    Parameters
    ----------
    sampler : selene_sdk.samplers.Sampler
        The sampler from which to draw data.

    Attributes
    ----------
    sampler : selene_sdk.samplers.Sampler
        The sampler from which to draw data.
    """
    def __init__(self, sampler):
        super(_SamplerDataset, self).__init__()
        self.sampler = sampler

    def __getitem__(self, index):
        """
        Retrieve sample(s) from self.sampler. Only index length affects the
        number of samples. The index values are not used.

        Parameters
        ----------
        index : int or any object with __len__ method implemented
            The size of index is used to determine the number of the
            samples to return.

        Returns
        ----------
        sequences, targets : tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the numeric representation of the
            sequence examples and their corresponding labels. The
            shape of `sequences` will be
            :math:`I \\times L \\times N`, where :math:`I` is
            `index`, :math:`L` is the sequence length, and
            :math:`N` is the size of the sequence type's alphabet.
            The shape of `targets` will be :math:`I \\times T`,
            where :math:`T` is the number of targets predicted.
        """
        sequences, targets, inds = self.sampler.sample(
            batch_size=1 if isinstance(index, int) else len(index))
        if sequences.shape[0] == 1:
            sequences = sequences[0,:]
            targets = targets[0,:]
        return sequences, targets, inds

    def __len__(self):
        """
        Implementing __len__ is required by the DataLoader. So as a workaround,
        this returns `sys.maxsize` which is a large integer which should
        generally prevent the DataLoader from reaching its size limit.

        Another workaround that is implemented is catching the StopIteration
        error while calling `next` and reinitialize the DataLoader.
        """
        return sys.maxsize


class SamplerDataLoader(DataLoader):
    """
    A DataLoader that provides parallel sampling for any `Sampler` object.
    `SamplerDataLoader` can be used with `MultiSampler` by specifying
    the `SamplerDataLoader` object as `train_sampler`, `validate_sampler`
    or `test_sampler` when initiating a `MultiSampler`.

    Parameters
    ----------
    sampler : selene_sdk.samplers.Sampler
        The sampler from which to draw data.
    num_workers : int, optional
        Default to 1. Number of workers to use for DataLoader.
    batch_size : int, optional
        Default to 1. The number of samples the iterator returns in one step.
    seed : int, optional
        Default to 436. The seed for random number generators.

    Attributes
    ----------
    sampler : selene_sdk.samplers.Sampler
        The sampler from which to draw data. Specified by the `sampler` param.
    num_workers : int
        Number of workers to use for DataLoader.
    batch_size : int
        The number of samples the iterator returns in one step.

    """
    def __init__(self,
                 sampler,
                 num_workers=1,
                 batch_size=1,
                 seed=436):
        def worker_init_fn(worker_id):
            """
            This function is called to initialize each worker with different
            numpy seeds (torch seeds are set by DataLoader automatically).
            """
            np.random.seed(seed + worker_id)

        args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "worker_init_fn": worker_init_fn
        }

        super(SamplerDataLoader, self).__init__(_SamplerDataset(sampler), **args)
        self.seed = seed


def unpackbits_sequence(sequence, s_len):
    sequence = np.unpackbits(sequence.astype(np.uint8), axis=-2)
    nulls = np.sum(sequence, axis=-1) == sequence.shape[-1]
    sequence = sequence.astype(float)
    sequence[nulls, :] = 1.0 / sequence.shape[-1]
    if sequence.ndim == 3:
        sequence = sequence[:, :s_len, :]
    else:
        sequence = sequence[:s_len, :]
    return sequence


def unpackbits_targets(targets, t_len):
    targets = np.unpackbits(targets, axis=-1).astype(float)
    if targets.ndim == 2:
        targets = targets[:, :t_len]
    else:
        targets = targets[:self.t_len]
    return targets


class _H5Dataset(Dataset):
    """
    This class provides a Dataset that directly loads sequences and targets
    from a hdf5 file. `_H5Dataset` is intended to be used internally by
    `H5DataLoader`.

    Parameters
    ----------
    file_path : str
        The file path of the hdf5 file.
    in_memory : bool, optional
        Default is False. If True, load entire dataset into memory.
    unpackbits : bool, optional
        Default is False. If True, unpack binary-valued array from uint8
        sequence and targets array. See `numpy.packbits` for details.
    sequence_key : str, optional
        Default is "sequences". Specify the name of the hdf5 dataset that contains
        sequence data.
    targets_key : str, optional
        Default is "targets". Specify the name of the hdf5 dataset that contains
        target data.

    Attributes
    ----------
    file_path : str
        The file path of the hdf5 file.
    in_memory : bool
        If True, load entire dataset into memory.
    unpackbits : bool
        If True, unpack binary-valued array from uint8
        sequence and targets array. See `numpy.packbits` for details.
    """
    def __init__(self,
                 file_path,
                 in_memory=False,
                 unpackbits=False,  # implies unpackbits for both
                 unpackbits_seq=False,
                 unpackbits_tgt=False,
                 sequence_key="sequences",
                 targets_key="targets",
                 indicators_key=False,
                 use_seq_len=None,
                 shift=False):
        super(_H5Dataset, self).__init__()
        self.file_path = file_path
        self.in_memory = in_memory

        self.unpackbits = unpackbits
        self.unpackbits_seq = unpackbits_seq
        self.unpackbits_tgt = unpackbits_tgt

        self.use_seq_len = use_seq_len
        self.shift = shift
        self._seq_start, self._seq_end = None, None

        self._initialized = False
        self._sequence_key = sequence_key
        self._targets_key = targets_key
        self._indicators_key = indicators_key

    def init(func):
        # delay initialization to allow multiprocessing
        @wraps(func)
        def dfunc(self, *args, **kwargs):
            if not self._initialized:
                self.db = h5py.File(self.file_path, 'r')
                key = 'indicator'
                if key not in self.db and self._indicators_key:
                    key = 'indicators'

                if self.unpackbits:
                    self.s_len = self.db['{0}_length'.format(self._sequence_key)][()]
                    self.t_len = self.db['{0}_length'.format(self._targets_key)][()]
                elif self.unpackbits_seq:
                    self.s_len = self.db['{0}_length'.format(self._sequence_key)][()]
                elif self.unpackbits_tgt:
                    self.t_len = self.db['{0}_length'.format(self._targets_key)][()]

                if self.in_memory:
                    self.sequences = np.asarray(self.db[self._sequence_key])
                    self.targets = np.asarray(self.db[self._targets_key])
                    self.indicators = None
                    if self._indicators_key:
                        self.indicators = np.asarray(self.db[key])
                else:
                    self.sequences = self.db[self._sequence_key]
                    self.targets = self.db[self._targets_key]
                    self.indicators = None
                    if self._indicators_key:
                        self.indicators = self.db[key]
                self._initialized = True
            return func(self, *args, **kwargs)
        return dfunc

    @init
    def __getitem__(self, index):
        if isinstance(index, int):
            index = index % self.sequences.shape[0]
        sequence = self.sequences[index]
        targets = self.targets[index]

        if self.unpackbits:
            sequence = unpackbits_sequence(sequence, self.s_len)
            targets = unpackbits_targets(targets, self.t_len)
        elif self.unpackbits_seq:
            sequence = unpackbits_sequence(sequence, self.s_len)
        elif self.unpackbits_tgt:
            targets = unpackbits_targets(targets, self.t_len)

        if self._seq_start is None:
            self._seq_start = 0
            self._seq_end = len(sequence)

            if self.use_seq_len is not None:
                mid = (self._seq_end - self._seq_start) // 2
                self._seq_start = mid - self.use_seq_len // 2
                self._seq_end = int(mid + np.ceil(self.use_seq_len / 2))
                if self.shift:
                    diff = (len(sequence) - self.use_seq_len) // 2
                    direction = np.random.choice([-1, 1])
                    shift = np.random.choice(np.arange(diff+1))
                    self._seq_start = self._seq_start + direction * shift
                    self._seq_end = self._seq_end + direction * shift
        sequence = sequence[self._seq_start:self._seq_end]

        # UNET ONLY #
        #targets = targets[:, self._seq_start:self._seq_end]
        #if np.random.randint(2) == 1:
        #    sequence = np.flip(sequence, axis=-1)
        #    targets = np.flip(targets, axis=-1)
        # UNET ONLY #
        if self.indicators is not None:
            return (torch.from_numpy(sequence.astype(np.float32)),
                    torch.from_numpy(targets.astype(np.float32)),
                    self.indicators[index])
        else:
            return (torch.from_numpy(sequence.astype(np.float32)),
                    torch.from_numpy(targets.astype(np.float32)),)


    @init
    def __len__(self):
        return self.sequences.shape[0]


class H5DataLoader(DataLoader):
    """
    H5DataLoader provides optionally parallel sampling from a HDF5
    dataset that contains sequences and targets data. The name of the
    array of sequences and targets data are specified by `sequence_key`
    and `targets_key` respectively. The sequences array should be
    of shape:math:`B \\times L \\times N`, where :math:`B` is
    the sample size, :math:`L` is the sequence length, and :math:`N` is
    the size of the sequence type's alphabet. The shape of the targets array
     will be :math:`B \\times F`, where :math:`F` is the number of features.

    H5DataLoader also supports compressed binary data (using `numpy.packbits`)
    with the `unpackbits` option. To generate compressed binary data, the
    sequences and targets array have to both be binary-valued, and then
    packed in the :math:`L` (sequence length) and `F` (number of features)
    dimensions, respectively.
    For the sequences array, represent unknown bases ("N"s) by binary
    data with all-ones in the encoding - they will be transformed to
    the correct representations in selene_sdk.sequences.Genome when unpacked.
    In addition, to unpack correctly, the length of the packed dimensions,
    i.e. :math:`L` and :math:`F` must be provided in two integer scalars
    named `{sequence_key}_length` and `{targets_key}_length` in the HDF5 file
    if `unpackbits==True`.

    You can generate a HDF5 dataset file by sampling from an online sampler
    using the script that we provided `scripts/write_sampled_h5.py`. An
    example config file template is in `config_examples/sample_h5.yml`.

    `H5DataLoader` can be used with `MultiSampler` by passing
    `SamplerDataLoader` object as `train_sampler`, `validate_sampler` or
    `test_sampler` when initiating a `MultiSampler`.

    Parameters
    ----------
    file_path : str
        The file path of the hdf5 file.
    in_memory : bool, optional
        Default is False. If True, load entire dataset into memory.
    num_workers : int, optional
        Default is 1. If greater than 1, use multiple processes to parallelize data
        sampling.
    use_subset : int, (int, int), list(int), or None, optional
        Default is None. If a single integer value is provided, sample from only
        the first `use_subset` rows of the dataset. If a tuple of integers is
        provided, `(<start-index>, <end-index>)`, sample from the range of
        rows specified. If a list of integers is provided, restrict to sampling
        only the indices specified in the list. If None, use the entire dataset.
    batch_size : int, optional
        Default is 1. Specify the batch size of the DataLoader.
    shuffle : bool, optional
        Default is True. If False, load the data in the original order.
    unpackbits : bool, optional
        Default is False. If True, unpack binary-valued array from uint8
        sequence and targets array. See `numpy.packbits` for details.
    sequence_key : str, optional
        Default is "sequences". Specify the name of the hdf5 dataset that contains
        sequence data.
    targets_key : str, optional
        Default is "targets". Specify the name of the hdf5 dataset that contains
        target data.

    Attributes
    ----------
    dataset : `_H5Dataset`
        The `_H5Dataset` to load data from.

    """
    def __init__(self,
                 dataset,
                 num_workers=1,
                 use_subset=None,
                 batch_size=1,
                 seed=436,
                 sampler=None,
                 batch_sampler=None,
                 shuffle=True):
        g = torch.Generator()
        g.manual_seed(seed)

        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            print("Worker seed", worker_seed)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        args = {
            "batch_size": batch_size,
            "pin_memory": True,
            "worker_init_fn": worker_init_fn,
            "sampler": sampler,
            "batch_sampler": batch_sampler,
            "generator": g,
        }

        if hasattr(dataset, 'in_memory'):
            args['num_workers'] = 0 if dataset.in_memory else num_workers
        else:
            args['num_workers'] = num_workers

        if use_subset is not None:
            from torch.utils.data.sampler import SubsetRandomSampler
            if isinstance(use_subset, int):
                use_subset = list(range(use_subset))
            elif isinstance(use_subset, tuple) and len(use_subset) == 2:
                use_subset = list(range(use_subset[0], use_subset[1]))
            args["sampler"] = SubsetRandomSampler(use_subset)
        else:
            args["shuffle"] = shuffle
        super(H5DataLoader, self).__init__(dataset, **args)

