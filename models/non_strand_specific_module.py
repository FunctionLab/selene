import numpy as np
import torch
from torch.nn.modules import Module


def flip(x, dim):
    """Reverses the elements in a given dimension `dim` of the Tensor.

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
    def __init__(self, model, mode="mean"):
        super(NonStrandSpecific, self).__init__()

        print(mode)
        self.model = model

        if mode != "mean" and mode != "max":
            raise ValueError(f"Mode should be one of 'mean' or 'max' but was"
                             "{mode!r}.")
        self.mode = mode

    def forward(self, input):

        reverse_input = flip(
            flip(input, 1), 2)

        output = self.model.forward(input)
        output_from_rev = self.model.forward(
            reverse_input)
        if self.mode == "mean":
            return (output + output_from_rev) / 2
        else:
            max_output = torch.max(
                output.abs(), output_from_rev.abs())
            np_output = output.data.cpu().numpy()
            print(np_output)

            it = np.nditer(np_output, flags=["multi_index"])
            while not it.finished:
                index = it.multi_index
                print(it[0])
                if max_output.data[index] != abs(it[0]):
                    max_output.data[index] = output_from_rev.data[index]
                else:
                    max_output.data[index] = it[0]
                it.iternext()
            return max_output

