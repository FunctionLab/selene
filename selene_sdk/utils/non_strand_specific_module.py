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
    mode : {'mean', 'max'}
        How to handle outputting a non-strand specific prediction.

    """

    def __init__(self, model, mode="mean"):
        super(NonStrandSpecific, self).__init__()

        self.model = model

        if mode != "mean" and mode != "max":
            raise ValueError("Mode should be one of 'mean' or 'max' but was"
                             "{0}.".format(mode))
        self.mode = mode
        self.from_lua = _is_lua_trained_model(model)

    def forward(self, input):
        reverse_input = None
        if self.from_lua:
            reverse_input = _flip(
                _flip(torch.squeeze(input, 2), 1), 2).unsqueeze_(2)
        else:
            reverse_input = _flip(_flip(input, 1), 2)

        output = self.model.forward(input)
        output_from_rev = self.model.forward(reverse_input)
        if type(output) == tuple:
            output1, output2 = output
            output_from_rev1, output_from_rev2 = output_from_rev
            if self.mode == "mean":
                return ((output1 + output_from_rev1) / 2,
                        (output2 + output_from_rev2) / 2)
            else:
                return (torch.max(output1, output_from_rev1),
                        torch.max(output2, output_from_rev2))
        else:
            if self.mode == "mean":
                return (output + output_from_rev) / 2
            else:
                return torch.max(output, output_from_rev)


