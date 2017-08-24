"""
Description:
    This script builds the model and trains it using user-specified input data.

Output:
    Saves model to a user-specified output file.

Usage:
    seq_model.py <genome-fa> <features-tabix> <features-file>
        <distinct-features> <holdout> <output-file>
        [--radius=<radius>]
        [--mode=<mode>]
        [-v | --verbose]
    seq_model.py -h | --help

Options:
    -h --help               Show this screen.

    <genome-fa>             The target organism's full genome sequence
                            FASTA file.
    <features-tabix>        The tabix-indexed sequence features .bed file.
    <features-file>         The non-tabix-indexed sequence features .bed file.
    <distinct-features>     The list of distinct features in the file.
    <holdout-chrs>          Specify which chromosomes should be in our holdout
                            set.
    <output-file>           The trained model will be saved to this file.

    --radius=<radius>       Specify the radius surrounding a target base.
                            (e.g. how big of a window/context is needed?)
                            [default: 100]
    --mode=<mode>           One of {"all", "train", "test"}
                            # TODO: should this be all vs. train/test?
                            # also, if you use this to call test you should
                            # expect an input of the trained model...
                            [default: "all"]
    -v --verbose            Logging information to stdout.
                            [default: False]
"""
import os
from time import time

from docopt import docopt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable


torch.set_num_threads(32)  # TODO: should this be a parameter?

# model specifications

N_FEATURES = 4
HEIGHT = 1
# WIDTH = # number of inputs or the window size...? TBD
WIDTH = 1000
N_KERNELS = [320, 480, 960]
N_CHANNELS = np.floor(  # account for window size and pooling layers
    (np.floor((WIDTH - 7.) / 4.) - 7.) / 4.) - 7
N_OUTPUTS = 919  # this is the number of chromatin features

class View(nn.Module):
    def __init__(self, *shape):
        super(nn.View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*shape)


deepsea = nn.modules.container.Sequential(

    nn.Conv2d(N_FEATURES, N_KERNELS[0], (8, 1), (1, 1), padding=0),
    nn.ReLU(inplace=True),  # should I use this flag?
    nn.MaxPool2d((4, 1), (4, 1)),
    nn.Dropout(p=0.2),  # why not have this be inplace?

    nn.Conv2d(N_KERNELS[0], N_KERNELS[1], (8, 1), (1, 1), padding=0),
    nn.ReLU(inplace=True),
    nn.MaxPool2d((4, 1), (4, 1)),
    nn.Dropout(p=0.2),

    nn.Conv2d(N_KERNELS[1], N_KERNELS[2], (8, 1), (1, 1), padding=0),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),

    View(N_KERNELS[3] * N_CHANNELS, 1),
    nn.Linear(N_KERNELS[3] * N_CHANNELS, N_OUTPUTS),
    nn.ReLU(inplace=True),
    nn.Linear(N_OUTPUTS, N_OUTPUTS),

    nn.Sigmoid())


if __name__ == "__main__":
    arguments = doctopt(
        __doc__,
        version="1.0")
    genome_fa_file = arguments["<genome-fa>"]
    features_tabix_file = arguments["<features-tabix>"]
    features_file = arguments["<features-file>"]
    # TODO: this should be in a file... especially when you get to hundreds of features
    features_list = list(arguments["<distinct-features>"])
    holdout = list(arguments["<holdout-chrs>"])
    output_file = arguments["<output-file>"]

    radius = int(arguments["--radius"])
    mode = arguments["--mode"]
    verbose = arguments["--verbose"]


hiddenSizes = [100,2]
n_lstm_layers = 2
rnn = nn.LSTM(input_size=4, hidden_size=hiddenSizes[0], num_layers=n_lstm_layers, batch_first=True, bidirectional=True)

conv = nn.modules.container.Sequential(
    nn.Conv1d(hiddenSizes[0]*2, hiddenSizes[0]*2, 1),
    nn.ReLU(),
    nn.Conv1d(hiddenSizes[0]*2, hiddenSizes[1], 1))

model = [rnn, conv]
useCuda = True
if useCuda:
    for module in model:
        module.cuda()

padding = (0, 0)
criterion = nn.MSELoss()
optimizer = [optim.SGD(module.parameters(), lr=0.05, momentum=0.95) for module in model]

def runBatch(batchSize=16, update=True, plot=False):
    window = sdata.radius * 2 + 1 + sum(padding)
    inputs = np.zeros((batchSize, window, len(BASES)))
    # should there be padding here?
    # also the '2' looks like it should be replaced with n_features
    targets = np.zeros((batchSize, sdata.radius * 2 + 1, 2))
    for i in range(batchSize):
        sequence, target = sdata.sample_mixture(0.5, padding=padding)
        inputs[i, :, :] = sequence
        #targets[i,:,:] = np.log10(target+1e-6)+6
        targets[i, :, :] = target  # score of just 1 ok?

    if useCuda:
        inputs = Variable(torch.Tensor(inputs).cuda(), requires_grad=True)
        targets = Variable(torch.Tensor(targets).cuda())
        h0 = Variable(torch.zeros(n_lstm_layers*2, batchSize, hiddenSizes[0]).cuda())
        c0 = Variable(torch.zeros(n_lstm_layers*2, batchSize, hiddenSizes[0]).cuda())
    else:
        inputs = Variable(torch.Tensor(inputs), requires_grad=True)
        targets = Variable(torch.Tensor(targets))
        h0 = Variable(torch.zeros(n_lstm_layers * 2, batchSize, hiddenSizes[0]))
        c0 = Variable(torch.zeros(n_lstm_layers * 2, batchSize, hiddenSizes[0]))

    outputs, hn = rnn(inputs, (h0, c0))
    outputs = conv(outputs.transpose(1,2)).transpose(1,2)

    loss = criterion(outputs,targets)

    if update:
        for module in model:
            module.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

    if plot:
        plt.figure()
        plt.plot(outputs.data.numpy().flatten(),targets.data.numpy().flatten(),'.',alpha=0.2)
        plt.show()
    return loss.data[0]


sdata = SplicingDataset(
    os.path.join(DIR, "hg38.fa"),
    os.path.join(DIR, "splicejunc.database.bed.sorted.gz"),
    os.path.join(DIR, "splicejunc.database.bed.sorted.gz"),
    ["5p", "3p"],
    ["chr8", "chr9"],
    radius=100,
    mode="train")


for _ in range(10000):
    sdata.train_mode()
    cumlossTrain = 0
    for _ in range(1000):
        cumlossTrain = cumlossTrain + runBatch()

    sdata.train_mode('test')
    cumlossTest = 0
    for _ in range(100):
        cumlossTest = cumlossTest + runBatch(update=False)
    print("Train loss: %.5f, Test loss: %.5f." % (cumlossTrain, cumlossTest) )


torch.save(model,os.path.join(DIR, "models/101bp.h100.cpu.model"))
