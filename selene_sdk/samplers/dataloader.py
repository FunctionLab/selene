"""
This module provides the `SamplerDataLoader` and  `SamplerDataSet` classes,
which allow parallel sampling for any Sampler or FileSampler using
torch DataLoader mechanism.
"""

import  sys

import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader

class _SamplerDataset(data.Dataset):
    """
    This class provides a Dataset interface for a Sampler or FileSampler. 
    `_SamplerDataset` is intended to be used with `SamplerDataLoader`.
    
    Parameters
    ----------
    sampler : selene_sdk.samplers.Sampler or 
        selene_sdk.samplers.file_samplers.FileSampler
        The sampler to draw data from.

    Attributes
    ----------
    sampler : selene_sdk.samplers.Sampler or 
        selene_sdk.samplers.file_samplers.FileSampler
        The sampler to draw data from.
    """
    def __init__(self, sampler):
        super(_SamplerDataset, self).__init__()
        self.sampler = sampler

    def __getitem__(self, index):
        """
        Retrieve sample(s) from self.sampler. Only index length affects the 
        size the samples. The index values are not used.

        Parameters
        ----------
        index : int or any object with __len__ method implemented
            The size of index is used to determined size of the samples 
            to return.

        Returns
        ----------
        sequences, targets : tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the numeric representation of the
            sequence examples and their corresponding labels. The
            shape of `sequences` will be
            :math:`B \\times L \\times N`, where :math:`B` is
            `batch_size`, :math:`L` is the sequence length, and
            :math:`N` is the size of the sequence type's alphabet.
            The shape of `targets` will be :math:`B \\times F`,
            where :math:`F` is the number of features.
        """
        sequences, targets = self.sampler.sample(batch_size=1 \
            if isinstance(index, int) else len(index))
        if sequences.shape[0] == 1:
            sequences = sequences[0,:]
            targets = targets[0,:]
        return sequences, targets

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
    A DataLoader that provides parallel sampling for any `Sampler`
    or `FileSampler` object. SamplerDataLoader requires sampler to be 
    initialized with `picklable=True` to enable multi-procesing.
    Currently `SamplerDataLoader` can be used with `MultiFileSampler` by
    passing `SamplerDataLoader` object as `train_sampler`, `validate_sampler`
    or `test_sampler` when initiating a `MultiFileSampler`.

    Parameters
    ----------
    sampler : selene_sdk.samplers.Sampler or selene_sdk.samplers.file_samplers.FileSampler
        The sampler to draw data from.
    num_workers : int, optional
        Default to 1. Number of workers to use for DataLoader.
    batch_size : int, optional
        Default to 1. The number of samples the iterator returns in one step.
    seed : int, optional
        Default to 436. The seed for random number generators.

    Attributes
    ----------
    dataset : selene_sdk.samplers.Sampler or selene_sdk.samplers.file_samplers.FileSampler
        The sampler to draw data from. Specified by the `sampler` argument.
    num_workers : int, optional
        Default to 1. Number of workers to use for DataLoader.
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
            
        super(SamplerDataLoader, self).__init__(_SamplerDataset(sampler),**args)


