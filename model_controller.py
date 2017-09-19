"""Execute the necessary steps to train the model
"""
import logging
import os
import shutil
from time import strftime, time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

from utils import AverageMeter


LOG = logging.getLogger("deepsea")
torch.set_num_threads(32)


class ModelController(object):
    def __init__(self, model, sampler,
                 loss_criterion, optimizer_args,
                 batch_size,
                 use_cuda=False, data_parallel=False,
                 prefix_outputs=None,
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
        self.optimizer = self._optimizer(**optimizer_args)

        self.batch_size = batch_size

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

        # TODO: validation set -> helper function
        n_validation_exs = 5000
        self._validation_data = []
        validation_targets = []
        self.sampler.set_mode("validate")
        n_validation_batches = int(n_validation_exs / self.batch_size)
        for _ in range(n_validation_batches):
            inputs, targets = self._run_batch()
            self._validation_data.append((inputs, targets))
            validation_targets.append(targets)
        self._all_validation_targets = np.vstack(validation_targets)
        LOG.info(
            ("Loaded {0} validation examples to evaluate after "
             "each training epoch.").format(n_validation_exs))

        self._training_loss = []
        self._validation_loss = []

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

    def _optimizer(self, lr=0.05, momentum=0.90, **kwargs):
        """Specify the optimizer to use. Here, it is stochastic gradient
        descent.

        TODO: Discuss where the optimizer should be specified
        [software design]
        """
        return torch.optim.SGD(self.model.parameters(),
                               lr=lr,
                               momentum=momentum,
                               **kwargs)

    def _run_batch(self):
        t_i_sampling = time()
        inputs = np.zeros((self.batch_size, self.sampler.window_size, 4))
        targets = np.zeros((self.batch_size, self.sampler.n_features))
        for i in range(self.batch_size):
            sequence, target = self.sampler.sample()
            inputs[i, :, :] = sequence
            targets[i, :] = np.any(target == 1, axis=0)
        t_f_sampling = time()
        LOG.debug(
            ("[BATCH] Time to sample {0} examples: {1} s.").format(
                 self.batch_size,
                 t_f_sampling - t_i_sampling))
        return (inputs, targets)

    def train_validate(self, n_epochs, n_train):
        """The training and validation process.
        Will sample (`n_epochs` * `n_train` * self.batch_size)
        positive and negative examples (total) in a single call
        to `train_validate`.

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
            ("[Train/validate] n_epochs: {0}, n_train: {1}, "
             "batch_size: {2}").format(
                n_epochs, n_train, self.batch_size))

        avg_batch_times = AverageMeter()
        min_loss = self.min_loss

        # for learning rate decay
        # TODO: gamma might need to be a parameter somewhere.
        scheduler = StepLR(self.optimizer, step_size=1, gamma=8e-7)

        for epoch in range(self.start_epoch, n_epochs):
            t_i = time()

            avg_losses_train = AverageMeter()
            avg_losses_validate = AverageMeter()
            cum_loss_train = 0.
            cum_loss_validate = 0.

            # training
            self.model.train()
            for _ in range(n_train):
                info = self.run_batch_training(
                    avg_losses_train,
                    avg_batch_times)
                cum_loss_train += info["loss"]
            LOG.debug("Ep {0} average training loss: {1}".format(
                epoch, avg_losses_train.avg))
            cum_loss_train /= n_train

            # evaluate using the validation set
            self.model.eval()
            collect_predictions = []
            for inputs, targets in self._validation_data:
                info = self._process(
                    inputs, targets,
                    avg_losses_validate,
                    avg_batch_times,
                    mode="validate")
                collect_predictions.append(info["predictions"])
                cum_loss_validate += info["loss"]
            all_predictions = np.vstack(collect_predictions)
            feature_aucs = []
            for index, feature_preds in enumerate(all_predictions.T):
                feature_targets = self._all_validation_targets[:, index]
                if len(np.unique(feature_targets)) > 1:
                    auc = roc_auc_score(feature_targets, feature_preds)
                    feature_aucs.append(auc)

            LOG.debug("[AUC] Average: {0}".format(np.average(feature_aucs)))
            print("[AUC] Average: {0}".format(np.average(feature_aucs)))

            LOG.debug("Ep {0} average validation loss: {1}".format(
                epoch, avg_losses_validate.avg))
            cum_loss_validate /= len(self._validation_data)

            t_f = time()
            LOG.info(
                ("[Train/validate] Epoch {0}: {1} s. "
                 "Training loss: {2}, validation loss: {3}.").format(
                     epoch, t_f - t_i, cum_loss_train, cum_loss_validate))

            scheduler.step()

            if epoch % 5 == 0:
                LOG.info(
                    ("[Train/validate] Average time to propagate a batch "
                     "of size {0} through the model: {1} s").format(
                        self.batch_size, avg_batch_times.avg))
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

        with open("vloss_" + self.prefix_outputs + ".txt",
                  mode="wt", encoding="utf-8") as \
                validation_loss_txt:
            validation_loss_txt.write(
                '\n'.join(str(loss) for loss in self._validation_loss))

        with open("tloss_" + self.prefix_outputs + ".txt",
                  mode="wt", encoding="utf-8") as \
                train_loss_txt:
            train_loss_txt.write(
                '\n'.join(str(loss) for loss in self._training_loss))

    def run_batch_training(self, avg_losses, avg_batch_times):
        """Create and process a training batch of positive/negative examples.

        Parameters
        ----------
        avg_losses : AverageMeter
            Used to track the average loss, within each epoch.
        avg_batch_times : AverageMeter
            Used to track the average time it takes to process a batch in the
            model, across all epochs.
        # TODO: REMOVE BELOW mode
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
        self.sampler.set_mode("train")  # just to make it explicit
        inputs, targets = self._run_batch()

        # logging information
        # TODO: helper function?
        proportion_features_in_inputs = []
        for i in range(self.batch_size):
            proportion_features_in_inputs.append(np.sum(targets[i, :]))
        proportion_features_in_inputs = np.array(
            proportion_features_in_inputs)
        proportion_features_in_inputs /= float(self.sampler.n_features)
        avg_prop_features = np.average(proportion_features_in_inputs)
        std_prop_features = np.std(proportion_features_in_inputs)

        count_features = np.sum(targets, axis=0)
        most_common_features = list(count_features.argsort())
        most_common_features.reverse()
        report_n_features = 10
        common_feats = {}
        for feature_index in most_common_features[:report_n_features]:
            feat = self.sampler.get_feature_from_index(feature_index)
            common_feats[feat] = count_features[feature_index]

        LOG.debug(
            ("[BATCH] proportion of features present in each example: "
             "avg {0}, std {1}").format(avg_prop_features,
                                        std_prop_features))
        LOG.debug(
            "[BATCH] {0} most common features present: {1}".format(
                report_n_features, common_feats))
        return self._process(
            inputs, targets, avg_losses, avg_batch_times, "train")

    def _process(self, inputs, targets,
                 avg_losses, avg_batch_times, mode):
        inputs = torch.Tensor(inputs)
        targets = torch.Tensor(targets)
        if self.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        if mode == "train":
            inputs = Variable(inputs)
            targets = Variable(targets)
        else:
            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True)

        t_i = time()
        output = self.model(inputs.transpose(1, 2))
        loss = self.criterion(output, targets)

        if mode == "train":
            LOG.debug("Updating the model after a training batch.")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_losses.update(loss.data[0], inputs.size(0))
        avg_batch_times.update(time() - t_i)

        # returns logging information
        log_info = {"batch_time": avg_batch_times.val,
                    "batch_time_avg": avg_batch_times.avg,
                    "loss": avg_losses.val,
                    "loss_avg": avg_losses.avg}
        if mode == "validate":
            log_info["predictions"] = output.data.cpu().numpy()
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
