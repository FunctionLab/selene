"""
DanQ architecture (Quang & Xie, 2016).
"""
import math

import numpy as np
import torch
import torch.nn as nn


class DanQ(nn.Module):
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
            Input sequence length
        n_genomic_features : int
            Total number of features to predict
        """
        super(DanQ, self).__init__()
        self.nnet = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=26),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=13, stride=13),
            nn.Dropout(0.2))

        self.bdlstm = nn.Sequential(
            nn.LSTM(
                320, 320, num_layers=1, batch_first=True, bidirectional=True))

        self._n_channels = math.floor(
            (sequence_length - 25) / 13)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._n_channels * 640, 925),
            nn.ReLU(inplace=True),
            nn.Linear(925, n_genomic_features),
            nn.Sigmoid())

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.nnet(x)
        reshape_out = out.transpose(0, 1).transpose(0, 2)
        out, _ = self.bdlstm(reshape_out)
        out = out.transpose(0, 1)
        reshape_out = out.contiguous().view(
            out.size(0), 640 * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict

def criterion():
    return nn.BCELoss()

def get_optimizer(lr):
    return (torch.optim.RMSprop, {"lr": lr})
