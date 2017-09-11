"""
Description:
    This script builds the model and trains it using user-specified input data.

Output:
    Saves model to a user-specified output file.

Usage:
    seq_model.py <genome-fa> <features-file> <features-gz>
        <uniq-features-file> <chrs-file> <output-file>
        [--holdout-chrs=<chrs>]
        [--radius=<radius>] [--window=<window-size>]
        [--random-seed=<rseed>]
        [--mode=<mode>] [-v | --verbose] [--use-cuda]
    seq_model.py -h | --help

Options:
    -h --help               Show this screen.

    <genome-fa>             The target organism's full genome sequence
                            FASTA file.
    <features-file>         The non-tabix-indexed sequence features .bed file.
    <features-gz>           The tabix-indexed sequence features .bed.gz file.
    <uniq-features-file>    INS DESCRIPTION
    <chrs-file>             INS DESCRIPTION
    <output-file>           The trained model will be saved to this file.

    --holdout-chrs=<chrs>   Specify which chromosomes should be in our holdout
                            set.
                            [default: chr8,chr9]
    --radius=<radius>       Specify the radius surrounding a target base.
                            A bin of length radius + 1 target base + radius
                            is annotated with a genomic features vector
                            based on the features file.
                            [default: 100]
    --window=<window-size>  Specify the input sequence window size.
                            The window is larger than the bin to provide
                            context for the bin sequence.
                            [default: 1001]
    --random-seed=<rseed>   Set the random seed.
                            [default: 123]
    --mode=<mode>           One of {"all", "train", "test"}
                            # TODO: should this be all vs. train/test?
                            # also, if you use this to call test you should
                            # expect instead to have a trained model as input
                            [default: all]
    -v --verbose            Logging information to stdout.
                            [default: False]
    --use-cuda              Whether CUDA is available to use or not.
                            [default: False]
"""
from time import time

from docopt import docopt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

from seqmodel import BASES
from seqmodel import Sampler

torch.set_num_threads(32)  # TODO: should this be a parameter?

# model specifications
# TODO: this would be great as a config file. e.g. model_specifications.py
N_FEATURES = 4
HEIGHT = 1
# WIDTH = # number of inputs or the window size...? TBD
WIDTH = 1001
N_KERNELS = [320, 480, 960]
#N_CHANNELS = int(np.floor(  # account for window size and pooling layers
#    (np.floor((WIDTH - 7.) / 4.) - 7.) / 4.) - 7)
#N_OUTPUTS = 381  # this is the number of chromatin features
N_OUTPUTS = 460
N_CHANNELS = int(np.floor(  # account for window size and pooling layers
    np.floor(WIDTH / 4.) / 4.))

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        print(self.shape)
        print(x.size())
        print(x.view(*self.shape).size())
        return x.view(*self.shape)


BATCH_SIZE = 96
deepsea = nn.modules.container.Sequential(

    #nn.Conv2d(N_FEATURES, N_KERNELS[0], (8, 1), (1, 1), padding=0),
    nn.Conv1d(N_FEATURES, N_KERNELS[0], 1),
    nn.ReLU(inplace=True),  # should I use this flag?
    # nn.MaxPool2d((4, 1), (4, 1)),
    nn.MaxPool1d(4),
    nn.Dropout(p=0.2),  # why not have this be inplace?

    #nn.Conv2d(N_KERNELS[0], N_KERNELS[1], (8, 1), (1, 1), padding=0),
    nn.Conv1d(N_KERNELS[0], N_KERNELS[1], 1),
    nn.ReLU(inplace=True),
    #nn.MaxPool2d((4, 1), (4, 1)),
    nn.MaxPool1d(4),
    nn.Dropout(p=0.2),

    #nn.Conv2d(N_KERNELS[1], N_KERNELS[2], (8, 1), (1, 1), padding=0),
    nn.Conv1d(N_KERNELS[1], N_KERNELS[2], 1),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),

    # TODO: the 16 is the batch size. Need to *not* hard code that...
    View((BATCH_SIZE, N_KERNELS[2] * N_CHANNELS)),
    nn.Linear(N_KERNELS[2] * N_CHANNELS, N_OUTPUTS),
    nn.ReLU(inplace=True),
    nn.Linear(N_OUTPUTS, N_OUTPUTS),

    nn.Sigmoid())


def runBatch(sampler, optimizers, window_size, use_cuda, batch_size=16, update=True, plot=False):
    inputs = np.zeros((batch_size, window_size, len(BASES)))
    targets = np.zeros((batch_size, sampler.n_features))
    for i in range(batch_size):
        sequence, target = sampler.sample_mixture()
        inputs[i, :, :] = sequence
        targets[i, :] = np.any(target == 1, axis=0)

    if use_cuda:
        inputs = Variable(torch.Tensor(inputs).cuda(), requires_grad=True)
        targets = Variable(torch.Tensor(targets).cuda())
    else:
        inputs = Variable(torch.Tensor(inputs), requires_grad=True)
        targets = Variable(torch.Tensor(targets))
    outputs = deepsea(inputs.transpose(1,2))
    loss = criterion(outputs, targets)

    if update:
        for module in model:
            module.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
    return loss.data[0]


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")
    genome_fa_file = arguments["<genome-fa>"]
    features_file = arguments["<features-file>"]
    features_gz_file = arguments["<features-gz>"]
    distinct_features = arguments["<uniq-features-file>"]
    chrs_list_txt = arguments["<chrs-file>"]
    output_file = arguments["<output-file>"]

    holdout = arguments["--holdout-chrs"].split(",")
    radius = int(arguments["--radius"])
    window_size = int(arguments["--window"])
    random_seed = int(arguments["--random-seed"])
    mode = arguments["--mode"]

    verbose = arguments["--verbose"]
    use_cuda = arguments["--use-cuda"]


    model = [deepsea]
    if use_cuda:
        for module in model:
            module.cuda()

    criterion = nn.BCELoss()
    optimizers = [optim.SGD(module.parameters(), lr=0.05, momentum=0.95) for module in model]

    ti = time()
    sampler = Sampler(
        genome_fa_file,
        features_gz_file,
        features_file,
        distinct_features,
        chrs_list_txt,
        holdout,
        radius=radius,
        window_size=window_size,
        random_seed=random_seed,
        mode=mode)

    n_epochs = 100
    #n_epochs = 2
    n_train = 8
    n_test = 2
    for i in range(n_epochs):
        ti_epoch = time()
        sampler.set_mode("train")
        cum_loss_train = 0
        for _ in range(n_train):
            cum_loss_train = cum_loss_train + runBatch(
                sampler, optimizers, window_size, use_cuda,
                batch_size=BATCH_SIZE)

        sampler.set_mode("test")
        cum_loss_test = 0
        for _ in range(n_test):
            cum_loss_test = cum_loss_test + runBatch(
                sampler, optimizers, window_size, use_cuda,
                update=False,
                batch_size=BATCH_SIZE)
        tf_epoch = time()
        print("Train loss: {0}, Test loss: {1}. Time: {2} s.".format(
            cum_loss_train / n_train, cum_loss_test / n_test,
            tf_epoch - ti_epoch))
        print("EPOCH NUMBER {0}".format(i))


    torch.save(model, output_file)
    tf = time()
    print("Took {0} to train and test this model.".format(tf - ti))
