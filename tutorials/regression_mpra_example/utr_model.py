"""
Model derived from "Human 5′ UTR design and variant effect prediction from a massively parallel translation assay", https://doi.org/10.1101/310375
"""

import torch
import numpy
import torch.nn as nn


class UTRModel(nn.Module):
    def __init__(self, sequence_length=50, n_targets=1):
        super(UTRModel, self).__init__()
        self.sequence_length = sequence_length
        kernel_size = 9 # Slight modification from model used in manuscript.
        n_filters = 120
        nodes = 40
        padding = 4 # Note that this will be slightly different from original model's "same" padding in Keras.
        self.cnn = nn.Sequential(
                nn.Conv1d(4, n_filters, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True))
        with torch.no_grad():
            tmp = torch.zeros(1, 4, self.sequence_length)
            dnn_input_size = self.cnn.forward(tmp).view(1, -1).shape[1]
            del tmp
        self.dnn = nn.Sequential(nn.Linear(dnn_input_size, nodes),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.20),
                                 nn.Linear(nodes, n_targets))
        # Copy weight initialization from Keras.
        def init_weight(x):
            if isinstance(x, nn.Linear) or isinstance(x, nn.Conv1d):
                if isinstance(x, nn.Linear):
                    fan_avg = (x.in_features + x.out_features) * 0.5
                else:
                    fan_avg = (x.weight.shape[0] + x.weight.shape[1]) * x.weight.shape[2] * 0.5
                limit = numpy.sqrt(3 / fan_avg)
                nn.init.uniform_(x.weight, -1 * limit, limit)
                x.bias.data.fill_(0)
        self.cnn.apply(init_weight)
        self.dnn.apply(init_weight)

    def forward(self, input):
        batch_size = input.shape[0]
        ret = self.dnn.forward(self.cnn.forward(input).view(batch_size, -1))
        return ret


def criterion():
    """
    The loss function to be optimized.
    """
    return nn.MSELoss(reduction='elementwise_mean')


def get_optimizer(lr):
    """
    The optimizer and parameters.
    """
    return (torch.optim.Adam, {"lr": lr, "betas": (0.9, 0.999), "eps": 1e-08})

