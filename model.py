"""The sequence-level model that learns to make predictions about what
features are present in a genomic sequence.
"""
import logging
import os
import shutil
from time import strftime, time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import AverageMeter


LOG = logging.getLogger("deepsea")
torch.set_num_threads(32)


class ModelController(object):
    def __init__(self, model, sampler,
                 loss_criterion, optimizer_args,
                 use_cuda=False, data_parallel=False):
        """Methods to train, validate, and test a PyTorch model.

        Parameters
        ----------
        model : torch.nn.Module
        sampler : Sampler
        loss_criterion : torch.nn._Loss
        optimizer_args : dict
        use_cuda : bool, optional
            Default is False.
        data_parallel : bool, optional
            Default is False.

        Attributes
        ----------
        model : torch.nn.Module
        sampler : Sampler
        criterion : torch.nn._Loss
        optimizer : torch.optim
        use_cuda : bool
        data_parallel : bool
        """
        self.model = model
        self.sampler = sampler
        self.criterion = loss_criterion
        self.optimizer = self._optimizer(**optimizer_args)
        self.use_cuda = use_cuda
        self.data_parallel = data_parallel

        if self.data_parallel:
            self.model = nn.DataParallel(model)
            LOG.debug("Wrapped model in DataParallel")

        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()
            LOG.debug("Set modules to use CUDA")

    def _optimizer(self, lr=0.5, momentum=0.95, **kwargs):
        """Specify the optimizer to use. Here, it is stochastic gradient
        descent.

        TODO: Discuss where the optimizer should be specified
        [software design]
        """
        return torch.optim.SGD(self.model.parameters(),
                               lr=lr,
                               momentum=momentum,
                               **kwargs)

    def train_validate(self,
                       n_epochs=1000,
                       batch_size=96,
                       n_train=8,
                       n_validate=2):
        """The training and validation process.
        Defaults will sample approximately 1 million positive and negative
        examples (total) in a single call to `train_validate`.

        Parameters
        ----------
        n_epochs : int, optional
            Default is 1000. The number of epochs in which we train
            and validate the model. Each epoch is a full "training cycle,"
            after which we update the weights of the model.
        batch_size : int, optional
            Default is 128. The number of samples to propagate through the
            network in one epoch.
        n_train : int, optional
            Default is 6. The `n_train` and `n_validation` parameters are used
            to maintain the proportion of train/validation examples we would
            like to sample in a single epoch. (Training, validation, and
            testing sets are completely separate from each other. By these
            defaults, we maintain a 60-20-20 percent division in the
            proportion of the full dataset represented during
            training-validation-testing, respectively.
        n_validation : int, optional
            Default is 2. See notes in `n_validation`.

        Returns
        -------
        None
        """
        avg_batch_times = AverageMeter()
        min_loss = float("inf")
        for epoch in range(n_epochs):
            t_i = time()
            avg_losses = AverageMeter()
            cum_loss_train = 0.
            for _ in range(n_train):
                info = self.run_batch(
                    batch_size, avg_batch_times, avg_losses, mode="train")
                cum_loss_train += info["loss"]
                # LOGGING MESSAGE
            print("train")
            print(info)
            cum_loss_train /= n_train

            cum_loss_validate = 0.
            for _ in range(n_validate):
                info = self.run_batch(
                    batch_size, avg_batch_times, avg_losses, mode="validate")
                cum_loss_validate += info["loss"]
                # LOGGING MESSAGE

            print("validate")
            print(info)
            cum_loss_validate /= n_validate
            t_f = time()
            LOG.debug(
                ("Epoch {0}: {1} s. "
                 "Training loss: {2}, validation loss: {3}.").format(
                     epoch, t_f - t_i, cum_loss_train, cum_loss_validate))

            is_best = cum_loss_train < min_loss
            min_loss = min(cum_loss_train, min_loss)

            print(self.model.__class__)
            print(self.model.__class__.__name__)
            self._save_checkpoint({
                "epoch": epoch,
                "arch": "DeepSEA",
                "state_dict": self.model.state_dict(),
                "min_loss": min_loss,
                "optimizer": self.optimizer.state_dict()}, is_best)

    def run_batch(self, batch_size, avg_batch_times, avg_losses, mode="train"):
        """Process a batch of positive/negative examples. Will update a model
        if the mode is set to "train" only.

        Parameters
        ----------
        batch_size : int
            Specify the batch size.
        avg_batch_times : AverageMeter
            Used to track the average time it takes to process a batch in the
            model, across all epochs.
        avg_losses : AverageMeter
            Used to track the average loss, within each epoch.
        mode : {"all", "train", "validate", "test"}, optional
            Default is "train".

        Returns
        -------
        dict()
            Information about the current batch time, average batch time,
            current loss, and average loss.
            { "batch_time" : float,
              "batch_time_avg" : float,
              "loss" : float,
              "loss_avg" : float }
        """
        self.sampler.set_mode(mode)

        inputs = np.zeros((batch_size, self.sampler.window_size, 4))
        targets = np.zeros((batch_size, self.sampler.n_features))

        t_i = time()
        for i in range(batch_size):
            sequence, target = self.sampler.sample_mixture()
            inputs[i, :, :] = sequence
            targets[i, :] = np.any(target == 1, axis=0)

        inputs = torch.Tensor(inputs)
        targets = torch.Tensor(targets)
        if self.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        inputs = Variable(inputs, requires_grad=True)
        targets = Variable(targets)

        output = self.model(inputs.transpose(1, 2))
        loss = self.criterion(output, targets)

        avg_losses.update(loss.data[0], inputs.size(0))

        if mode == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_batch_times.update(time() - t_i)
        t_i = time()

        # returns logging information
        return {"batch_time": avg_batch_times.val,
                "batch_time_avg": avg_batch_times.avg,
                "loss": avg_losses.val,
                "loss_avg": avg_losses.avg}

    @staticmethod
    def _save_checkpoint(state, is_best,
                         dir_path=None,
                         filename="checkpoint.pth.tar"):
        """Saves snapshot of the model state to file.

        Parameters
        ----------
        state : dict
            Information about the state of the model
        is_best : bool
            Is this the model's best performance so far?
        dir_path : str, optional
            Default is None. Will output file to the current working directory
            if no path to directory is specified.
        filename : str, optional
            Default is "checkpoint.pth.tar". Specify the checkpoint filename.

        Returns
        -------
        None
        """
        if dir_path is None:
            dir_path = os.getcwd()
        time_str = strftime("%Y%m%d_%H%M")
        cp_filepath = os.path.join(
            dir_path, "{0}_{1}".format(time_str, filename))
        torch.save(state, cp_filepath)
        if is_best:
            best_filepath = os.path.join(dir_path,
                "{0}_model_best.pth.tar".format(time_str))
            shutil.copyfile(cp_filepath, best_filepath)


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


def deepsea(filepath=None, **kwargs):
    """Initializes a new (untrained) DeepSEA model or loads
    a trained model from a filepath.

    Parameters
    ----------
    filepath : str, optional
        Default is None.
    **kwargs : dict
        Parameters should match those of the DeepSEA constructor
        (window_size, n_genomic_features)

    Returns
    -------
    DeepSEA
    """
    model = DeepSEA(**kwargs)
    if filepath is not None:
        model.load_state_dict(torch.load(filepath))
        model.eval()
    return model
