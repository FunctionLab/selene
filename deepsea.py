"""DeepSEA architecture (Zhou & Troyanskaya, 2015)
"""
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
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Dropout(p=0.2),

            nn.Conv1d(320, 480, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        self.n_channels = int(np.floor(np.floor(
            window_size / 4.) / 4.))

        self.classifier = nn.Sequential(
            nn.Linear(960 * self.n_channels, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),

            nn.Sigmoid())

    def forward(self, x):
        """Forward propagation of a batch.
        """
        x = self.conv_net(x)
        x = x.view(x.size(0), 960 * self.n_channels)
        x = self.classifier(x)
        return x


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
