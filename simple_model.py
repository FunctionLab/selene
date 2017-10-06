"""A simple model used for testing.
Note this is Linear + Sigmoid (where the Sigmoid is
built into the criterion I'm using,
BCEWithLogitsLoss). I've tried it with Sigmoid in
the model + BCELoss and could not observe a significant difference
in performance.
"""
import math

import numpy as np
import torch
import torch.nn as nn


class DeepSEA(nn.Module):
    def __init__(self, window_size, n_genomic_features):
        """

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
        self.linear = nn.Linear(4 * window_size, n_genomic_features)


    def forward(self, x):
        """Forward propagation of a batch.
        """
        #for m in self.children():
        #    print(m)
        #    print(m.weight)
        #    m.weight.data.renorm_(2, 1, 0.9)
        # 128 x 4 x 1001
        (B, H, W) = x.data.size()
        x = x.contiguous().view(B, H * W)
        out = self.linear(x)
        return out


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
