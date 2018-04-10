"""Execute the necessary steps to train the model
"""
import logging
import math
import os
import shutil
import sys
from time import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau


logger = logging.getLogger("selene")


def initialize_logger(out_filepath, verbosity=1, stdout_handler=False):
    """This function can only be called successfully once.
    If the logger has already been initialized with handlers,
    the function exits. Otherwise, it proceeds to set the
    logger configurations.
    """
    logger = logging.getLogger("selene")
    # check if logger has already been initialized
    if len(logger.handlers):
        return

    if verbosity == 0:
        logger.setLevel(logging.WARN)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    elif verbosity == 2:
        logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handle = logging.FileHandler(out_filepath)
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)

    if stdout_handler:
        stream_handle = logging.StreamHandler(sys.stdout)
        stream_handle.setFormatter(formatter)
        logger.addHandler(stream_handle)


class ModelController(object):

    def __init__(self,
                 model,
                 data_sampler,
                 loss_criterion,
                 optimizer_class,
                 optimizer_args,
                 batch_size,
                 max_steps,
                 report_metrics_every_n_steps,
                 output_dir,
                 n_validation_samples,
                 save_checkpoint=1000,
                 cpu_n_threads=32,
                 use_cuda=False,
                 data_parallel=False,
                 checkpoint_resume=None):
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
        self.sampler = data_sampler
        self.criterion = loss_criterion
        self.optimizer = optimizer_class(
            self.model.parameters(), **optimizer_args)

        self.batch_size = batch_size
        self.max_steps = max_steps
        self.nth_step_report_metrics = report_metrics_every_n_steps
        self.save_checkpoint = save_checkpoint

        torch.set_num_threads(cpu_n_threads)

        self.use_cuda = use_cuda
        self.data_parallel = data_parallel
        self.output_dir = output_dir
        #initialize_logger(os.path.join(output_dir, "{0}.log".format(__name__)))

        if self.data_parallel:
            self.model = nn.DataParallel(model)
            logger.debug("Wrapped model in DataParallel")

        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()
            logger.debug("Set modules to use CUDA")

        self._create_validation_set(n_validation_samples)

        self.start_step = 0
        self.min_loss = float("inf")
        if checkpoint_resume is not None:
            self.start_step = checkpoint_resume["step"]
            self.min_loss = checkpoint_resume["min_loss"]
            self.optimizer.load_state_dict(
                checkpoint_resume["optimizer"])
            logger.info(
                ("Resuming from checkpoint: "
                 "step {0}, min loss {1}").format(
                    self.start_step, self.min_loss))

        self.training_loss = []
        self.nth_step_stats = {
            "validation_loss": [],
            "auc": []
        }

    def _create_validation_set(self, n_validation_samples):
        """Used in `__init__`.
        """
        t_i = time()
        self._validation_data, self._all_validation_targets = \
            self.sampler.get_validation_set(
                self.batch_size, n_samples=n_validation_samples)
        print(len(self._validation_data), len(self._all_validation_targets))
        t_f = time()
        logger.info(("{0} s to load {1} validation examples ({2} validation "
                     "batches) to evaluate after each training step.").format(
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

    def train_and_validate(self):
        """The training and validation process.
        """
        logger.info(
            ("[TRAIN] max_steps: {0}, batch_size: {1}").format(
                self.max_steps, self.batch_size))

        min_loss = self.min_loss
        scheduler = ReduceLROnPlateau(
            self.optimizer, 'max', patience=16, verbose=True,
            factor=0.8)
        for step in range(self.start_step, self.max_steps):
            train_loss = self.train()
            self.training_loss.append(train_loss)

            # @TODO: if step and step % ...
            if step % self.nth_step_report_metrics == 0:
                validation_loss, auc = self.validate()
                self.nth_step_stats["validation_loss"].append(validation_loss)
                self.nth_step_stats["auc"].append(auc)
                scheduler.step(math.ceil(auc * 1000.0) / 1000.0)

                is_best = validation_loss < min_loss
                min_loss = min(validation_loss, min_loss)
                self._save_checkpoint({
                    "step": step,
                    "arch": self.model.__class__.__name__,
                    "state_dict": self.model.state_dict(),
                    "min_loss": min_loss,
                    "optimizer": self.optimizer.state_dict()}, is_best)
                logger.info(
                    ("[METRICS] step={0}: "
                     "Training loss: {1}, validation loss: {2}.").format(
                        step, train_loss, validation_loss))

            if step % self.save_checkpoint == 0:
                self._save_checkpoint({
                    "step": step,
                    "arch": self.model.__class__.__name__,
                    "state_dict": self.model.state_dict(),
                    "min_loss": min_loss,
                    "optimizer": self.optimizer.state_dict()}, False)

    def train(self):
        """Create and process a training batch of positive/negative examples.
        """
        self.model.train()
        self.sampler.set_mode("train")
        inputs, targets = self._get_batch()

        inputs = torch.Tensor(inputs)
        targets = torch.Tensor(targets)

        if self.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs)
        targets = Variable(targets)

        predictions = self.model(inputs.transpose(1, 2))
        loss = self.criterion(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        """
        training_loss = None
        def closure():
            predictions = self.model(inputs.transpose(1, 2))
            loss = self.criterion(predictions, targets)
            loss.backward()
            training_loss = loss.data[0]
            return loss

        self.optimizer.zero_grad()
        self.optimizer.step(closure)
        """
        return loss.data[0]

    def validate(self):
        self.model.eval()

        validation_losses = []
        collect_predictions = []
        for (inputs, targets) in self._validation_data:
            inputs = torch.Tensor(inputs)
            targets = torch.Tensor(targets)

            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True)

            predictions = self.model(inputs.transpose(1, 2))
            validation_loss = self.criterion(
                predictions, targets).data[0]

            collect_predictions.append(predictions.data.cpu().numpy())
            validation_losses.append(validation_loss)
        all_predictions = np.vstack(collect_predictions)
        #print(all_predictions.shape)
        feature_aucs = []
        for index, feature_preds in enumerate(all_predictions.T):
            feature_targets = self._all_validation_targets[:, index]
            if len(np.unique(feature_targets)) > 1:
                auc = roc_auc_score(feature_targets, feature_preds)
                feature_aucs.append(auc)
        logger.debug("[METRICS] average AUC: {0}".format(np.average(feature_aucs)))
        print("[VALIDATE] average AUC: {0}".format(np.average(feature_aucs)))

        self.nth_step_stats["auc"].append(np.average(feature_aucs))
        return np.average(validation_losses), np.average(feature_aucs)

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
        logger.info("[TRAIN] {0}: Saving model state to file.".format(
            state["step"]))
        cp_filepath = os.path.join(
            self.output_dir, filename)
        torch.save(state, cp_filepath)
        if is_best:
            best_filepath = os.path.join(
                self.output_dir,
                "best_model.pth.tar")
            shutil.copyfile(cp_filepath, best_filepath)
