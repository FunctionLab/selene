"""DeepSEA architecture (Zhou & Troyanskaya, 2015).
"""
import math

import torch
import torch.nn as nn

from flatten import Flatten

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes,
                               kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4,
                               kernel_size=7, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Block(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9,
                stride=stride, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=9,
                stride=1, padding=4, bias=False),
            nn.BatchNorm1d(out_channels))
        self.downsample = downsample

    def forward(self, x):
        """
        residual = x
        out = self.net(x)
        if self.downsample:
            residual = self.downsample(x)
            out += residual
            out = self.relu(out)
        return out
        """
        residual = x
        #print(x.size())
        out = self.net(x)
        #print(x.size())
        if self.downsample is not None:
            residual = self.downsample(x)
            #print("DS {0}".format(residual.size()))
        out += residual
        #print("CONCAT {0}".format(out.size()))
        out = self.relu(out)
        #print("END {0}".format(out.size()))
        return out

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
        n_blocks = [4, 3, 2, 2, 2]
        layer = [64, 128, 240, 128, n_genomic_features]
        #layer = [64, 128, 320, 640, n_genomic_features]
        self.inplanes = layer[0]
        self.init_layer = nn.Sequential(
            nn.Conv1d(4, layer[0], kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(layer[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.blocks1 = nn.Sequential(
            self._make_layer(layer[0], 2, expansion=1, stride=1),
            self._make_layer(layer[1], 2, expansion=1, stride=2))
        self.blocks2 = nn.Sequential(
            self._make_layer(layer[1], 2, expansion=1, stride=2),
            self._make_layer(layer[2], 2, expansion=1, stride=2),
            self._make_layer(layer[3], 2, expansion=1, stride=2),
            self._make_layer(layer[4], 2, expansion=1, stride=2),
            self._make_layer(layer[4], 2, expansion=1, stride=1))

        #self.avg_pool = nn.AvgPool1d(7, stride=1)
        third_dim = int(math.ceil(window_size / 128.))# - 7 + 1
        #third_dim = int(math.ceil(window_size / 640.))# - 7 + 1
        self.flatten = Flatten(layer[4], third_dim, n_genomic_features)
        self.sigmoid = nn.Sigmoid()
        self._weight_initialization()

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                #m.weight.data.normal_(0, 0.05)
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, expansion=1, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * expansion))
        layers = []
        layers.append(Block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(Block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_layer(x) # 128, 64, 251
        #print("DS: {0}".format(x.size()))
        x = self.blocks1(x)
        #print("PB1: {0}".format(x.size()))
        x = self.blocks2(x)
        #print("PB2: {0}".format(x.size()))
        #x = self.blocks3(x)
        #print("PB3: {0}".format(x.size()))

        #x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        #print(x.size())

        x = self.flatten(x)
        x = self.sigmoid(x)
        return x

def criterion():
    return nn.BCELoss()

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

