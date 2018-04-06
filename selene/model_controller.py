"""Execute the necessary steps to train the model
"""
import logging
import math
import os
import shutil
from time import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils import AverageMeter


logger = logging.getLogger("selene")
torch.set_num_threads(32)


class ModelController(object):

    def __init__(self, model, sampler,
                 loss_criterion,
                 optimizer_class, optimizer_args,
                 batch_size,
                 n_steps_per_epoch,
                 output_dir,
                 checkpoint_resume=None,
                 use_cuda=False, data_parallel=False):
        """Methods to train and validate a PyTorch model.

        Parameters
        ----------
        model : torch.nn.Module
        sampler : Sampler
        loss_criterion : torch.nn._Loss
        optimizer_args : dict
        batch_size : int
            Specify the batch size to process examples. Should be a power of 2.
        use_cuda : bool, optional
            Default is False. Specify whether CUDA is available for torch
            to use during training.
        data_parallel : bool, optional
            Default is False. Specify whether multiple GPUs are available
            for torch to use during training.
        prefix_outputs : str, optional
            Default is None. If None, prefix output files (e.g. the latest
            checkpoint and the best performing state of the model)
            with the current day.
        checkpoint_resume : torch.save object, optional
            Default is None. If `checkpoint_resume` is not None, assumes
            the input is a model saved via `torch.save` that can be
            loaded to resume training.

        Attributes
        ----------
        model : torch.nn.Module
        sampler : Sampler
        criterion : torch.nn._Loss
        optimizer : torch.optim
        batch_size : batch_size
        use_cuda : bool
        data_parallel : bool
        prefix_outputs : str
        """
        self.model = model
        self.sampler = sampler
        self.criterion = loss_criterion
        self.optimizer = optimizer_class(
            self.model.parameters(), **optimizer_args)

        self.batch_size = batch_size
        self.n_steps_per_epoch = n_steps_per_epoch

        self.use_cuda = use_cuda
        self.data_parallel = data_parallel

        self.output_dir = output_dir

        if self.data_parallel:
            self.model = nn.DataParallel(model)
            logger.debug("Wrapped model in DataParallel")

        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()
            logger.debug("Set modules to use CUDA")

        self._create_validation_set()

        self.start_epoch = 0
        self.min_loss = float("inf")
        if checkpoint_resume is not None:
            self.start_epoch = checkpoint_resume["epoch"]
            self.min_loss = checkpoint_resume["min_loss"]
            self.optimizer.load_state_dict(
                checkpoint_resume["optimizer"])
            logger.info(
                ("Resuming from checkpoint: "
                 "epoch {0}, min loss {1}").format(
                    self.start_epoch, self.min_loss))

        self.stats = {
            "training_loss": [],
            "validation_loss": [],
            "AUC": []
        }

    def _create_validation_set(self):
        """Used in `__init__`.
        """
        t_i = time()
        self._validation_data, self._all_validation_targets = \
            self.sampler.get_validation_set(self.batch_size, n_samples=32000)
        t_f = time()
        logger.info(("{0} s to load {1} validation examples ({2} validation "
                     "batches) to evaluate after each training epoch.").format(
                      t_f - t_i,
                      len(self._validation_data) * self.batch_size,
                      len(self._validation_data)))

    def _get_batch(self):
        """Sample `self.batch_size` times. Return inputs and targets as a
        batch.
        """
        t_i_sampling = time()
        batch_sequences, batch_targets = self.sampler.sample(
            batch_size=self.batch_size)
        t_f_sampling = time()
        logger.debug(
            ("[BATCH] Time to sample {0} examples: {1} s.").format(
                 self.batch_size,
                 t_f_sampling - t_i_sampling))
        return (batch_sequences, batch_targets)

    def train_and_validate(self, n_epochs):
        """The training and validation process.
        Will sample (`n_epochs` * `n_train` * self.batch_size)
        examples in total.

        Parameters
        ----------
        n_epochs : int
            The number of epochs in which we train and validate the model.
            Each epoch is a full "training cycle," after which we update the
            weights of the model.
        n_train : int
            The number of training batches to process in a single epoch.

        Returns
        -------
        None
        """
        logger.info(
            ("[TRAIN/VALIDATE] n_epochs: {0}, n_train: {1}, "
             "batch_size: {2}").format(
                n_epochs, self.n_steps_per_epoch, self.batch_size))

        min_loss = self.min_loss

        scheduler = ReduceLROnPlateau(
            self.optimizer, 'max', patience=16, verbose=True,
            factor=0.8)
        for epoch in range(self.start_epoch, n_epochs):
            t_i = time()
            train_loss_avg = self.train(epoch)
            validate_loss_avg = self.validate(epoch)
            self.stats["training_loss"].append(train_loss_avg)
            self.stats["validation_loss"].append(validate_loss_avg)
            auc_avg = self.stats["AUC"][-1]

            t_f = time()

            logger.info(
                ("[EPOCH] {0}: {1} s. "
                 "Training loss: {2}, validation loss: {3}.").format(
                     epoch, t_f - t_i, train_loss_avg, validate_loss_avg))

            is_best = validate_loss_avg < min_loss
            min_loss = min(validate_loss_avg, min_loss)

            scheduler.step(math.ceil(auc_avg * 1000.0) / 1000.0)

            logger.info(
                "[EPOCH] {0}: Saving model state to file.".format(epoch))
            self._save_checkpoint({
                "epoch": epoch,
                "arch": self.model.__class__.__name__,
                "state_dict": self.model.state_dict(),
                "min_loss": min_loss,
                "optimizer": self.optimizer.state_dict()}, is_best)

    def train(self, epoch):
        avg_losses_train = AverageMeter()
        self.model.train()

        for batch_number in range(self.n_steps_per_epoch):
            self.run_batch_training(avg_losses_train, batch_number)
        logger.debug("[TRAIN] Ep {0} average training loss: {1}".format(
            epoch, avg_losses_train.avg))
        return avg_losses_train.avg

    def validate(self, epoch):
        avg_losses_validate = AverageMeter()
        self.model.eval()

        collect_predictions = []
        for batch_number, (inputs, targets) in \
                enumerate(self._validation_data):
            info = self._pass_through_model_validate(
                inputs, targets, batch_number,
                avg_losses_validate)
            collect_predictions.append(info["predictions"])

        logger.debug("[VALIDATE] Ep {0} average validation loss: {1}".format(
            epoch, avg_losses_validate.avg))

        all_predictions = np.vstack(collect_predictions)
        feature_aucs = []
        for index, feature_preds in enumerate(all_predictions.T):
            feature_targets = self._all_validation_targets[:, index]
            if len(np.unique(feature_targets)) > 1:
                auc = roc_auc_score(feature_targets, feature_preds)
                feature_aucs.append(auc)
        logger.debug("[AUC] Average: {0}".format(np.average(feature_aucs)))
        print("[AUC] average: {0}".format(np.average(feature_aucs)))

        self.stats["AUC"].append(np.average(feature_aucs))
        return avg_losses_validate.avg

    def run_batch_training(self, avg_losses, batch_number):
        """Create and process a training batch of positive/negative examples.

        Parameters
        ----------
        avg_losses : AverageMeter
            Used to track the average loss, within each epoch.
        batch_number : int
            The current batch number. Used for monitoring/logging.

        Returns
        -------
        dict()
            Information about [TODO: revise] the current batch time,
            average batch time, current loss, and average loss.
            { "batch_time" : float,
              "batch_time_avg" : float,
              "loss" : float,
              "loss_avg" : float }
        """
        self.sampler.set_mode("train")
        inputs, targets = self._get_batch()
        return self._pass_through_model_train(
            inputs, targets, batch_number, avg_losses)

    def _pass_through_model_train(self, inputs, targets, batch_number,
                                  avg_losses):
        t_i = time()
        inputs = torch.Tensor(inputs)
        targets = torch.Tensor(targets)

        if self.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs)
        targets = Variable(targets)

        def closure():
            batch_output = self.model(inputs.transpose(1, 2))
            loss = self.criterion(batch_output, targets)
            loss.backward()
            avg_losses.update(loss.data[0], inputs.size(0))
            return loss

        logger.debug("Updating the model after a training batch.")
        self.optimizer.zero_grad()
        self.optimizer.step(closure)

        batch_time = time() - t_i

        log_info = {"batch_time": batch_time,
                    "loss": avg_losses.val,
                    "loss_avg": avg_losses.avg}
        return log_info

    def _pass_through_model_validate(self, inputs, targets, batch_number,
                                     avg_losses):
        t_i = time()

        inputs = torch.Tensor(inputs)
        targets = torch.Tensor(targets)

        if self.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)

        t_i = time()
        output = self.model(inputs.transpose(1, 2))
        loss = self.criterion(output, targets)

        avg_losses.update(loss.data[0], inputs.size(0))
        batch_time = time() - t_i

        log_info = {"batch_time": batch_time,
                    "loss": avg_losses.val,
                    "loss_avg": avg_losses.avg,
                    "predictions": output.data.cpu().numpy()}
        return log_info

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
        cp_filepath = os.path.join(
            self.output_dir, filename)
        torch.save(state, cp_filepath)
        if is_best:
            best_filepath = os.path.join(
                self.output_dir,
                "best_model.pth.tar")
            shutil.copyfile(cp_filepath, best_filepath)
