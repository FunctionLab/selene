"""DeepSEA architecture (Zhou & Troyanskaya, 2015).
"""
import math

import numpy as np
import torch
import torch.nn as nn


class DeepSEA(nn.Module):
    def __init__(self, window_size, n_genomic_features):
        """The DeepSEA architecture.

        Parameters
        ----------
        window_size : int
        n_genomic_features : int

        Attributes
        ----------
        conv_net : torch.nn.Sequential
        n_channels : int
        classifier : torch.nn.Sequential
        """
        super(DeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            np.floor(
                (np.floor(
                    (window_size - reduce_by) / pool_kernel_size)
                - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(960 * self.n_channels, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid())

        self._weight_initialization()

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.05)

    def forward(self, x):
        """Forward propagation of a batch.
        """
        for layer in self.conv_net.children():
            if isinstance(layer, nn.Conv1d):
                layer.weight.data.renorm_(2, 1, 0.9)
        for layer in self.classifier.children():
            if isinstance(layer, nn.Linear):
                layer.weight.data.renorm_(2, 1, 0.9)
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self.n_channels)
        predict = self.classifier(reshape_out)
        return predict


def deepsea(window_size, n_genomic_features, filepath=None):
    """Initializes a new (untrained) DeepSEA model or loads
    a trained model from a filepath.

    Parameters
    ----------
    window_size : int
        The window size is the input sequence length for a single
        training example.
    n_genomic_features : int
        The number of genomic features (classes) to predict.
    filepath : str, optional
        Default is None.

    [TODO] Note this function has not been tested.

    Returns
    -------
    DeepSEA
    """
    model = DeepSEA(window_size, n_genomic_features)
    if filepath is not None:
        model.load_state_dict(torch.load(filepath))
        model.eval()
    return model
