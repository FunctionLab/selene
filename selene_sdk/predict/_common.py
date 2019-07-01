"""
Prediction specific utility functions.
"""
import math

import numpy as np
import torch
from torch.autograd import Variable

from ..utils import _is_lua_trained_model


def get_reverse_complement(allele, complementary_base_dict):
    """
    Get the reverse complement of the input allele.

    Parameters
    ----------
    allele : str
        The sequence allele
    complementary_base_dict : dict(str)
        The dictionary that maps each base to its complement

    Returns
    -------
    str
        The reverse complement of the allele.

    """
    if allele == '*' or allele == '-' or len(allele) == 0:
        return '*'
    a_complement = []
    for a in allele:
        a_complement.append(complementary_base_dict[a])
    return ''.join(list(reversed(a_complement)))


def predict(model, batch_sequences, use_cuda=False):
    """
    Return model predictions for a batch of sequences.

    Parameters
    ----------
    model : torch.nn.Sequential
        The model, on mode `eval`.
    batch_sequences : numpy.ndarray
        `batch_sequences` has the shape :math:`B \\times L \\times N`,
        where :math:`B` is `batch_size`, :math:`L` is the sequence length,
        :math:`N` is the size of the sequence type's alphabet.
    use_cuda : bool, optional
        Default is `False`. Specifies whether CUDA-enabled GPUs are available
        for torch to use.

    Returns
    -------
    numpy.ndarray
        The model predictions of shape :math:`B \\times F`, where :math:`F`
        is the number of features (classes) the model predicts.

    """
    inputs = torch.Tensor(batch_sequences)
    if use_cuda:
        inputs = inputs.cuda()
    with torch.no_grad():
        inputs = Variable(inputs)

        if _is_lua_trained_model(model):
            outputs = model.forward(inputs.transpose(1, 2).unsqueeze_(2))
        else:
            outputs = model.forward(inputs.transpose(1, 2))
        return outputs.data.cpu().numpy()


def _pad_sequence(sequence, to_length, unknown_base):
    diff = (to_length - len(sequence)) / 2
    pad_l = int(np.floor(diff))
    pad_r = math.ceil(diff)
    sequence = ((unknown_base * pad_l) + sequence + (unknown_base * pad_r))
    return str.upper(sequence)


def _truncate_sequence(sequence, to_length):
    start = int((len(sequence) - to_length) // 2)
    end = int(start + to_length)
    return str.upper(sequence[start:end])
