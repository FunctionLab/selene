"""
Description:
    This script builds the model and trains it using user-specified input data.

Output:
    Saves model to a user-specified output file.

Usage:
    seq_model.py <genome-fa> <features-file> <features-gz> <output-file>
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

from seqmodel import Sampler
from model import DeepSEA, SeqModel

if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")
    genome_fa_file = arguments["<genome-fa>"]
    features_file = arguments["<features-file>"]
    features_gz_file = arguments["<features-gz>"]
    output_file = arguments["<output-file>"]

    holdout = arguments["--holdout-chrs"].split(",")
    radius = int(arguments["--radius"])
    window_size = int(arguments["--window"])
    random_seed = int(arguments["--random-seed"])
    mode = arguments["--mode"]

    verbose = arguments["--verbose"]
    use_cuda = arguments["--use-cuda"]


    model = DeepSEA(window_size, 381)

    criterion = nn.BCELoss()

    ti = time()
    sampler = Sampler(
        genome_fa_file,
        features_file,
        features_gz_file,
        holdout,
        radius=radius,
        window_size=window_size,
        random_seed=random_seed,
        mode=mode)


    runner = SeqModel(model, sampler, criterion, {}, use_cuda=True, data_parallel=False)
    runner.train_validate(n_epochs=2, n_train=10, n_validate=3)

    tf = time()
    print("Took {0} to train and test this model.".format(tf - ti))
