"""Execute the necessary steps to train the model
"""
import logging
import os
import shutil
from time import time

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.io
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from utils import AverageMeter


LOG = logging.getLogger("deepsea")
torch.set_num_threads(32)


class ModelController(object):

    def __init__(self, model, samples_dir,
                 loss_criterion,
                 optimizer_class, optimizer_args,
                 batch_size,
                 n_train_batch_per_epoch,
                 output_dir,
                 checkpoint_resume=None,
                 use_cuda=False, data_parallel=False):
        self.model = model
        self.criterion = loss_criterion
        self.optimizer = optimizer_class(
            self.model.parameters(), **optimizer_args)

        self.batch_size = batch_size
        self.n_train_batch_per_epoch = n_train_batch_per_epoch

        self.use_cuda = use_cuda
        self.data_parallel = data_parallel

        self.output_dir = output_dir

        if self.data_parallel:
            self.model = nn.DataParallel(model)
            LOG.debug("Wrapped model in DataParallel")

        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()
            LOG.debug("Set modules to use CUDA")

        self.start_epoch = 0
        self.min_loss = float("inf")
        if checkpoint_resume is not None:
            self.start_epoch = checkpoint_resume["epoch"]
            self.min_loss = checkpoint_resume["min_loss"]
            self.optimizer.load_state_dict(
                checkpoint_resume["optimizer"])
            LOG.info(
                ("Resuming from checkpoint: "
                 "epoch {0}, min loss {1}").format(
                    self.start_epoch, self.min_loss))

        self.stats = {
            "training_loss": [],
            "validation_loss": [],
            "AUC": []
        }

        self._load_train_validate_mats(samples_dir)

    def _load_train_validate_mats(self, samples_dir):
        train_mat = h5py.File(
            os.path.join(samples_dir, "train.mat"), 'r')
        train_targets = train_mat["traindata"][()].astype(float).T
        n_train, n_features = train_targets.shape

        train_samples = train_mat["trainxdata"][()].T
        n_train, n_bases, seq_len = train_samples.shape

        valid_mat = scipy.io.loadmat(
            os.path.join(samples_dir, "valid.mat"))
        n_valid, n_features = valid_mat["validdata"].shape
        n_valid, n_bases, n_sequence = valid_mat["validxdata"].shape

        self.train_targets = train_targets
        self.train_samples = train_samples
        self.n_train = n_train

        self.n_features = n_features

        """
        # OVERFITTING TEST
        n_overfit = 32000

        self.train_targets = train_targets[:n_overfit, :]
        self.train_samples = train_samples[:n_overfit, :, :]
        self.n_train = n_overfit
        """

        self.valid_targets = valid_mat["validdata"].astype(float)
        self.valid_samples = valid_mat["validxdata"].astype(float)
        self.n_valid = n_valid

        """
        # OVERFITTING TEST
        self.valid_targets = self.valid_targets[:n_overfit - 1024, :]
        self.valid_samples = self.valid_samples[:n_overfit - 1024, :, :]
        self.n_valid = n_overfit - 1024
        """

        print("n_train: {0}, n_valid: {1}".format(n_train, n_valid))
        train_mat.close()

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
        LOG.info(
            ("[TRAIN/VALIDATE] n_epochs: {0}, n_train: {1}, "
             "batch_size: {2}").format(
                n_epochs, self.n_train_batch_per_epoch, self.batch_size))

        min_loss = self.min_loss

        # TODO: gamma might need to be a parameter somewhere.
        # learning rate decay.
        scheduler = StepLR(self.optimizer, step_size=18, gamma=0.05)

        for epoch in range(self.start_epoch, n_epochs):
            t_i = time()
            scheduler.step()

            train_indices = np.arange(self.n_train)
            np.random.shuffle(train_indices)

            valid_indices = np.arange(self.n_valid)
            np.random.shuffle(valid_indices)

            train_loss_avg = self.train(epoch, train_indices)
            validate_loss_avg = self.validate(epoch, valid_indices)

            self.stats["training_loss"].append(train_loss_avg)
            self.stats["validation_loss"].append(validate_loss_avg)
            auc_avg = self.stats["AUC"][-1]

            t_f = time()

            LOG.info(
                ("[EPOCH] {0}: {1} s. "
                 "Training loss: {2}, validation loss: {3}.").format(
                     epoch, t_f - t_i, train_loss_avg, validate_loss_avg))

            is_best = validate_loss_avg < min_loss
            min_loss = min(validate_loss_avg, min_loss)

            LOG.info(
                "[EPOCH] {0}: Saving model state to file.".format(epoch))
            self._save_checkpoint({
                "epoch": epoch,
                "arch": self.model.__class__.__name__,
                "state_dict": self.model.state_dict(),
                "min_loss": min_loss,
                "optimizer": self.optimizer.state_dict()}, is_best)

    def train(self, epoch, train_indices):
        avg_losses_train = AverageMeter()
        self.model.train()
        n_samples = 1000000
        #n_samples = self.n_train
        collect_predictions = []
        for i in range(0, n_samples, self.batch_size):
            use_indices = train_indices[i:i+self.batch_size]
            inputs = self.train_samples[use_indices, :, :]
            inputs = inputs.astype(float)
            inputs = np.transpose(inputs, (0, 2, 1))

            targets = self.train_targets[use_indices, :]

            info = self._pass_through_model_train(
                inputs, targets, int(i / self.batch_size), avg_losses_train)
            collect_predictions.append(info["predictions"])
        LOG.debug("[TRAIN] Ep {0} average training loss: {1}".format(
            epoch, avg_losses_train.avg))
        self._log_training_info(targets)

        all_predictions = np.vstack(tuple(collect_predictions))
        feature_aucs = []
        for index, feature_preds in enumerate(all_predictions.T):
            feature_targets = self.train_targets[train_indices[:n_samples], index]
            if len(np.unique(feature_targets)) > 1:
                auc = roc_auc_score(feature_targets, feature_preds)
                feature_aucs.append(auc)
        LOG.debug("[AUC] TRAIN: {0}".format(np.average(feature_aucs)))
        print("[AUC] TRAIN: {0}".format(np.average(feature_aucs)))
        return avg_losses_train.avg

    def validate(self, epoch, validate_indices):
        avg_losses_validate = AverageMeter()
        self.model.eval()

        collect_predictions = []

        for i in range(0, self.n_valid, self.batch_size):
            use_indices = validate_indices[i:i+self.batch_size]

            inputs = self.valid_samples[use_indices, :, :]
            inputs = np.transpose(inputs, (0, 2, 1))

            targets = self.valid_targets[use_indices, :]
            info = self._pass_through_model_validate(
                inputs, targets, int(i / self.batch_size),
                avg_losses_validate)
            collect_predictions.append(info["predictions"])

        LOG.debug("[VALIDATE] Ep {0} average validation loss: {1}".format(
            epoch, avg_losses_validate.avg))

        all_predictions = np.vstack(collect_predictions)
        feature_aucs = []
        for index, feature_preds in enumerate(all_predictions.T):
            feature_targets = self.valid_targets[validate_indices, index]
            if len(np.unique(feature_targets)) > 1:
                auc = roc_auc_score(feature_targets, feature_preds)
                feature_aucs.append(auc)
        LOG.debug("[AUC] Average: {0}".format(np.average(feature_aucs)))
        print("[AUC] average: {0}".format(np.average(feature_aucs)))

        self.stats["AUC"].append(np.average(feature_aucs))
        return avg_losses_validate.avg

    def _log_training_info(self, targets):
        proportion_features_in_inputs = np.sum(targets, axis=1)
        proportion_features_in_inputs /= float(self.n_features)
        avg_prop_features = np.average(proportion_features_in_inputs)
        std_prop_features = np.std(proportion_features_in_inputs)
        LOG.debug(
            ("[BATCH] proportion of features present in each example: "
             "avg {0}, std {1}").format(avg_prop_features,
                                        std_prop_features))

        count_features = np.sum(targets, axis=0)
        most_common_features = list(count_features.argsort())
        most_common_features.reverse()
        report_n_features = 10
        common_feats = {}
        for feature_index in most_common_features[:report_n_features]:
            common_feats[feature_index] = count_features[feature_index]
        LOG.debug(
            "[BATCH] {0} most common features present: {1}".format(
                report_n_features, common_feats))

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

        output = self.model(inputs.transpose(1, 2))
        loss = self.criterion(output, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        avg_losses.update(loss.data[0], inputs.size(0))

        LOG.debug("Updating the model after a training batch.")

        batch_time = time() - t_i

        log_info = {"batch_time": batch_time,
                    "loss": avg_losses.val,
                    "loss_avg": avg_losses.avg,
                    "predictions": output.data.cpu().numpy()}
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
