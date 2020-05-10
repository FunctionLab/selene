"""
HeartENN architecture (Richter et al., 2020).
"""
import numpy as np
import torch
import torch.nn as nn


class HeartENN(nn.Module):
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
            Length of sequence context on which to train.
        n_genomic_features : int
            The number of chromatin features to predict.

        Attributes
        ----------
        conv_net : torch.nn.Sequential
        classifier : torch.nn.Sequential

        """
        super(HeartENN, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 60, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(60, 60, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(60),

            nn.Conv1d(60, 80, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(80, 80, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(80),
            nn.Dropout(p=0.4),

            nn.Conv1d(80, 240, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(240, 240, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(240),
            nn.Dropout(p=0.6))

        reduce_by = 2 * (conv_kernel_size - 1)
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(240 * self._n_channels, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_genomic_features),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid())

    def forward(self, x):
        """Forward propagation of a batch.i

        """
        for layer in self.conv_net.children():
            if isinstance(layer, nn.Conv1d):
                layer.weight.data.renorm_(2, 0, 0.9)
        for layer in self.classifier.children():
            if isinstance(layer, nn.Linear):
                layer.weight.data.renorm_(2, 0, 0.9)
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 240 * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict

def criterion():
    return nn.BCELoss()

def get_optimizer(lr):
    return (torch.optim.SGD,
            {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})
