"""
This module provides the NonStrandSpecific class.
"""
import torch
from torch.nn.modules import Module

from . import _is_lua_trained_model


def _flip(x, dim):
    """
    Reverses the elements in a given dimension `dim` of the Tensor.

    source: https://github.com/pytorch/pytorch/issues/229
    """
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(
        x.size(0), x.size(1), -1)[:, getattr(
            torch.arange(x.size(1)-1, -1, -1),
            ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class NonStrandSpecific(Module):
    """
    A torch.nn.Module that wraps a user-specified model architecture if the
    architecture does not need to account for sequence strand-specificity.

    Parameters
    ----------
    model : torch.nn.Module
        The user-specified model architecture.
    mode : {'mean', 'max'}, optional
        Default is 'mean'. NonStrandSpecific will pass the input and the
        reverse-complement of the input into `model`. The mode specifies
        whether we should output the mean or max of the predictions as
        the non-strand specific prediction.

    Attributes
    ----------
    model : torch.nn.Module
        The user-specified model architecture.

    """

    def __init__(self, model, mode="mean"):
        super(NonStrandSpecific, self).__init__()

        self.model = model

        if mode == "mean":
            self.reduce_fn = lambda x, y: (x + y) / 2
        elif mode == "max":
            self.reduce_fn = torch.max
        else:
            raise ValueError("Mode should be one of 'mean' or 'max' but was"
                             "{0}.".format(mode))

        self.from_lua = _is_lua_trained_model(model)

    def _forward_input_with_reversed_sequence(self, input):
        multi_inputs = isinstance(input, dict)
        sequence = input if not multi_inputs else input["sequence_batch"]
        reversed_sequence = None
        if self.from_lua:
            reversed_sequence = _flip(
                _flip(torch.squeeze(sequence, 2), 1), 2).unsqueeze_(2)
        else:
            reversed_sequence = _flip(_flip(sequence, 1), 2)

        input_rev = None
        if multi_inputs:
            input_rev = input.copy()
            input_rev["sequence_batch"] = reversed_sequence
        else:
            input_rev = reversed_sequence

        return self.model.forward(input_rev)

    def forward(self, input):
        """Computes NN output for the given sequence and for a reversed sequence,
        applies `self.reduce_fn` function to those outputs, and returns the result.

        Parameters
        ----------
            input : numpy.ndarray or dict(str, numpy.ndarray)
                Model's inputs. Can be just a sequence or multi-inputs.

        """
        output = self.model.forward(input)
        output_from_rev = self._forward_input_with_reversed_sequence(input)

        return self.reduce_fn(output, output_from_rev)
