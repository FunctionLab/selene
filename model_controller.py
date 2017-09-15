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
                 use_cuda=False, data_parallel=False,
                 prefix_outputs=None,
                 checkpoint_resume=None):
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
        prefix_outputs : str, optional
            Default is None. If None, prefix output files (e.g. the latest
            checkpoint and the best performing state of the model)
            with the current day.

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
        self.prefix_outputs = prefix_outputs

        if self.data_parallel:
            self.model = nn.DataParallel(model)
            LOG.debug("Wrapped model in DataParallel")

        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()
            LOG.debug("Set modules to use CUDA")

        if self.prefix_outputs is None:
            self.prefix_outputs = strftime("%Y%m%d")

        self._training_loss = []
        self._validation_loss = []

        self.start_epoch = 0
        self.min_loss = float("inf")
        if checkpoint_resume is not None:
            self.start_epoch = checkpoint_resume["epoch"]
            self.min_loss = checkpoint_resume["min_loss"]
            self.optimizer.load_state_dict(
                checkpoint_resume["optimizer"])
            LOG.info("Checkpoint data: epoch {0}, min loss {1}".format(
                self.start_epoch, self.min_loss))

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
                       batch_size=128,
                       n_train=80,
                       n_validate=10):
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
            Default is 80. The `n_train` and `n_validation` parameters are used
            to maintain the proportion of train/validation examples we would
            like to sample in a single epoch. (Training, validation, and
            testing sets are completely separate from each other. By these
            defaults, we maintain a 80-10-10 percent division in the
            proportion of the full dataset represented during
            training-validation-testing, respectively.
        n_validation : int, optional
            Default is 10. See notes in `n_validation`.

        Returns
        -------
        None
        """
        avg_batch_times = AverageMeter()
        min_loss = self.min_loss

        for epoch in range(self.start_epoch, n_epochs):
            t_i = time()
            avg_losses_train = AverageMeter()
            cum_loss_train = 0.
            self.model.train()
            for _ in range(n_train):
                info = self.run_batch(
                    batch_size,
                    avg_batch_times,
                    avg_losses_train,
                    mode="train")
                cum_loss_train += info["loss"]
            LOG.debug("Ep {0} average training loss: {1}".format(
                epoch, avg_losses_train.avg))
            cum_loss_train /= n_train

            avg_losses_validate = AverageMeter()
            cum_loss_validate = 0.
            self.model.eval()
            for _ in range(n_validate):
                info = self.run_batch(
                    batch_size,
                    avg_batch_times,
                    avg_losses_validate,
                    mode="validate")
                cum_loss_validate += info["loss"]
            LOG.debug("Ep {0} average validation loss: {1}".format(
                epoch, avg_losses_validate.avg))
            cum_loss_validate /= n_validate
            t_f = time()
            LOG.info(
                ("Epoch {0}: {1} s. "
                 "Training loss: {2}, validation loss: {3}.").format(
                     epoch, t_f - t_i, cum_loss_train, cum_loss_validate))

            if epoch % 5 == 0:
                LOG.info(
                    ("Average time to propagate a batch of size {0} "
                     "through the model: {1} s").format(
                        batch_size, avg_batch_times.avg))
            self._training_loss.append(cum_loss_train)
            self._validation_loss.append(cum_loss_validate)

            is_best = cum_loss_validate < min_loss
            min_loss = min(cum_loss_validate, min_loss)

            LOG.info(
                "Saving model state to file on epoch {0}.".format(epoch))
            self._save_checkpoint({
                "epoch": epoch,
                "arch": self.model.__class__.__name__,
                "state_dict": self.model.state_dict(),
                "min_loss": min_loss,
                "optimizer": self.optimizer.state_dict()}, is_best)

        # Include a last log message if the last `epoch` was not a multiple
        # 5. e.g., we are interested in knowing the average time after the
        # model was trained over all epochs.
        if epoch % 5 != 0:
            LOG.info(
                ("Average time to propagate a batch of size {0} through the "
                 "model: {1} s").format(
                    batch_size, avg_batch_times.avg))

        with open("./validation_loss.txt", mode="wt", encoding="utf-8") as \
                validation_loss_txt:
            validation_loss_txt.write(
                '\n'.join(str(loss) for loss in self._validation_loss))

        with open("./train_loss.txt", mode="wt", encoding="utf-8") as \
                train_loss_txt:
            train_loss_txt.write(
                '\n'.join(str(loss) for loss in self._training_loss))

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
        n_features_in_inputs = []

        t_i_sampling = time()
        for i in range(batch_size):
            sequence, target = self.sampler.sample()
            inputs[i, :, :] = sequence
            targets[i, :] = np.any(target == 1, axis=0)
            n_features_in_inputs.append(np.sum(targets[i, :]))
        t_f_sampling = time()

        n_features_in_inputs = np.array(n_features_in_inputs)
        n_features_in_inputs /= float(self.sampler.n_features)
        avg_percent_features = np.average(n_features_in_inputs)
        std_percent_features = np.std(n_features_in_inputs)

        count_features = np.sum(targets, axis=0)
        most_common_features = list(count_features.argsort())
        most_common_features.reverse()
        report_n_features = 10
        common_feats = {}
        for feature_index in most_common_features[:report_n_features]:
            feat = self.sampler.get_feature_from_index(feature_index)
            common_feats[feat] = count_features[feature_index]

        LOG.debug(
            ("Proportion of features present in each example of batch: "
             "avg {0}, std {1}").format(avg_percent_features,
                                        std_percent_features))
        LOG.debug(
            "{0} most common features present in this batch: {1}".format(
                report_n_features, common_feats))

        inputs = torch.Tensor(inputs)
        targets = torch.Tensor(targets)
        if self.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        inputs = Variable(inputs, requires_grad=True)
        targets = Variable(targets)
        t_f_tensor_conv = time()
        LOG.debug(
            ("Time to sample {0} examples: {1} s. "
             "Time to convert to tensor format: {2} s.").format(
                 batch_size,
                 t_f_sampling - t_i_sampling,
                 t_f_tensor_conv - t_f_sampling))

        t_i = time()
        output = self.model(inputs.transpose(1, 2))
        loss = self.criterion(output, targets)

        avg_losses.update(loss.data[0], inputs.size(0))

        if mode == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_batch_times.update(time() - t_i)

        # returns logging information
        return {"batch_time": avg_batch_times.val,
                "batch_time_avg": avg_batch_times.avg,
                "loss": avg_losses.val,
                "loss_avg": avg_losses.avg}

    def _save_checkpoint(self, state, is_best,
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
        cp_filepath = os.path.join(
            dir_path, "{0}_{1}".format(self.prefix_outputs, filename))
        torch.save(state, cp_filepath)
        if is_best:
            best_filepath = os.path.join(
                dir_path,
                "{0}_model_best.pth.tar".format(self.prefix_outputs))
            shutil.copyfile(cp_filepath, best_filepath)
