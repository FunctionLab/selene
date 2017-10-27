"""
Description:
    This script builds the model and trains it using user-specified input data.

Output:
    Saves model to a user-specified output file.

Usage:
    seq_model.py <genome-fa> <features-data-gz> <feature-coords-data>
        <uniq-features> <output-file>
        [--radius=<radius>] [--window=<window-size>]
        [--random-seed=<rseed>]
        [--mode=<mode>] [--chrs-test=<chrs>] [--chrs-validate=<chrs>]
        [--n-epochs=<epochs>] [--batch-size=<batch>]
        [--log=<file-handle>] [-s | --stdout] [-v | --verbose]
        [--use-cuda] [--data-parallel]
    seq_model.py -h | --help

Options:
    -h --help               Show this screen.

    <genome-fa>             The target organism's full genome sequence
                            FASTA file.
    <features-data-gz>      Tabix-indexed features dataset. *.bed.gz file.
    <feature-coords-data>   A tab-separated (usually *.bed) file with the
                            genome coordinates of each feature.
                            Because of space/time concerns, this file omits
                            the feature (label) for each row and only contains
                            the columns [chr, start, end] in order. This is
                            sufficient for querying the actual feature in the
                            tabix-indexed file.
    <uniq-features>         *.txt file of the unique features in our dataset.
                            Each feature is on its own line.
    <output-file>           The trained model is saved to this file.

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

    --mode=<mode>           One of {"all", "train"}. "all" assumes that there
                            is no need to partition the input dataset into
                            train/validate/test sections.
                            For testing an already-trained model, please use
                            the script ./evaluate_model.py.
                            [default: train]
    --chrs-test=<chrs>      The chromosomes in our holdout test set.
                            Comma-separated, no spaces.
                            Manual check of the default tells us that we are
                            holding out 10% of our dataset for test.
                            [default: chr8,chr9]
    --chrs-validate=<chrs>  The chromosomes in our holdout validation set.
                            Comma-separated, no spaces.
                            Manual check of the default tells us that we are
                            holding out 10% of our dataset for validation.
                            [default: chr6,ch7]
    --n-epochs=<epochs>     The number of epochs
                            [default: 1000]
    --batch-size=<batch>    The number of training examples to propagation
                            through the model in one training iteration.
    --train-prop=<prop>     Specify the percentage of data not in the holdout
                            chromosomes that should be available for training.
                            (1 - <prop>) will be set aside for validation.
                            [default: 0.8]

    --log=<file-handle>     Output logging information to a file. Either
                            specify a filename of your own or use the default
                            filename.
                            [default: log_train.out]
    -s --stdout             Will also output logging information to stdout.
                            [default: False]
    -v --verbose            Include debug messages in logging information.
                            [default: False]

    --use-cuda              Whether CUDA is available to use or not.
                            [default: False]
    --data-parallel         Whether batch processing can be parallelized
                            over multiple GPUs
                            [default: False]
"""
import logging
import sys
from time import time

from docopt import docopt
import torch
from torch import nn

from deepsea import DeepSEA
#from model import DeepSEA
#from four_layer_model import DeepSEA
from model_controller import ModelController
from sampler import ChromatinFeaturesSampler

if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")

    genome_fa_file = arguments["<genome-fa>"]
    features_data_gz = arguments["<features-data-gz>"]
    feature_coords_data = arguments["<feature-coords-data>"]
    unique_features = arguments["<uniq-features>"]
    output_model = arguments["<output-file>"]

    radius = int(arguments["--radius"])
    window_size = int(arguments["--window"])
    random_seed = int(arguments["--random-seed"])

    mode = arguments["--mode"]
    holdout_test = arguments["--chrs-test"].split(",")
    holdout_validate = arguments["--chrs-validate"].split(",")
    n_epochs = int(arguments["--n-epochs"])
    batch_size = int(arguments["--batch-size"])

    output_log = arguments["--log"]
    to_stdout = arguments["--stdout"]
    verbose = arguments["--verbose"]

    use_cuda = arguments["--use-cuda"]
    data_parallel = arguments["--data-parallel"]

    log = logging.getLogger("deepsea")
    if verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    output_log = "{0}_{1}".format(
        str(time())[:10], output_log)
    file_handle = logging.FileHandler(output_log)
    file_handle.setFormatter(formatter)
    log.addHandler(file_handle)

    if to_stdout:
        stream_handle = logging.StreamHandler(sys.stdout)
        stream_handle.setFormatter(formatter)
        log.addHandler(stream_handle)

    t_i = time()
    sampler = ChromatinFeaturesSampler(
        genome_fa_file,
        features_data_gz,
        feature_coords_data,
        unique_features,
        holdout_test,
        holdout_validate,
        bin_radius=radius,
        window_size=window_size,
        random_seed=random_seed,
        mode=mode,
        sample_from="positive")
    #    sample_from="proportion",
    #    sample_positive_prop=0.75)

    t_i_model = time()
    model = DeepSEA(sampler.window_size, sampler.n_features)

    checkpoint = None
    resume = False
    if resume:
        log.info("Resuming training from checkpoint.")
        checkpoint = torch.load("20170914_model_best.pth.tar")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

    # TODO: would prefer to not have to import & specify this in the
    # train_model.py script, I think?
    criterion = nn.BCEWithLogitsLoss()
    optimizer_args = {"use_optim": "SGD", "lr": 0.045, "momentum": 0.9, "weight_decay": 5e-7}
    #optimizer_args = {"use_optim": "Adam", "lr": 0.04, "eps": 1e-6, "weight_decay": 5e-7}

    t_f_model = time()
    log.debug("Finished initializing the {0} model: {1} s".format(
        model.__class__.__name__, t_f_model - t_i_model))

    print(model)
    print(optimizer_args)

    runner = ModelController(
        model, sampler, criterion, optimizer_args,
        batch_size,
        prefix_outputs="{0}_LR{1}_BS{2}_EPS{3}".format(
            optimizer_args["use_optim"], optimizer_args["lr"],
            batch_size, n_epochs),
        use_cuda=use_cuda, data_parallel=data_parallel,
        checkpoint_resume=checkpoint)
    log.info("Training model: {0} epochs, {1} batch size.".format(
        n_epochs, batch_size))
    runner.train_and_validate(n_epochs, 2000)

    t_f = time()
    log.info("./train_model.py completed in {0} s.".format(t_f - t_i))
