"""The sequence-level model that learns to make predictions about what
features are present in a genomic sequence.
"""
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


N_BASES = 4


torch.set_num_threads(32)


class AverageMeter(object):
    """Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def run_batch(sampler, model, criterion,
              optimizer=None, mode="train",
              batch_size=16, use_cuda=False):
    batch_times = AverageMeter()
    losses = AverageMeter()

    sampler.set_mode(mode)

    inputs = np.zeros((batch_size, sampler.window_size, 4))
    targets = np.zeros((batch_size, sampler.n_features))

    t_i = time.time()
    for i in range(batch_size):
        sequence, target = sampler.sample_mixture()
        inputs[i, :, :] = sequence
        targets[i, :] = np.any(target == 1, axis=0)

    inputs = torch.Tensor(inputs)
    targets = torch.Tensor(targets)
    if use_cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()
    inputs = Variable(inputs, requires_grad=True)
    targets = Variable(targets)

    output = model(inputs.transpose(1, 2))
    loss = criterion(output, targets)

    losses.update(loss.data[0], inputs.size(0))

    if mode == "train" and optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    batch_times.update(time.time() - t_i)
    t_i = time.time()

    # returns logging information
    return {"batch_time": batch_times.val,
            "batch_time_avg": batch_times.avg,
            "loss": losses.val,
            "loss_avg": losses.avg}




class SeqModel(object):
    def __init__(self, model, sampler,
                 loss_criterion, optimizer_args,
                 use_cuda=False, data_parallel=False):
        self.model = model
        self.sampler = sampler
        self.criterion = loss_criterion
        self.optimizer = self._optimizer(**optimizer_args)

        if data_parallel:
            model = nn.DataParallel(model)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()

    def _optimizer(self, lr=0.5, momentum=0.95, **kwargs):
        return torch.optim.SGD(self.model.parameters(),
                               lr=lr,
                               momentum=momentum,
                               **kwargs)

    def train_validate(self, batch_size=16, n_epochs=10000, n_train=600,
                       n_validate=200):
        min_loss = float("inf")
        for epoch in range(n_epochs):
            cum_loss_train = 0.
            for _ in range(n_train):
                info = run_batch(self.sampler,
                                 self.model,
                                 self.criterion,
                                 self.optimizer,
                                 mode="train",
                                 batch_size=batch_size,
                                 use_cuda=self.use_cuda)
                cum_loss_train += info["loss"]
                # LOGGING MESSAGE
            cum_loss_train /= n_train

            cum_loss_validate = 0.
            for _ in range(n_validate):
                info = run_batch(self.sampler,
                                 self.model,
                                 self.criterion,
                                 self.optimizer,
                                 mode="validate",
                                 batch_size=batch_size,
                                 use_cuda=self.use_cuda)
                cum_loss_validate += info["loss"]
                # LOGGING MESSAGE
            cum_loss_train /= n_validate

            # LOGGING MESSAGE

            is_best = cum_loss_train < min_loss
            min_loss = min(cum_loss_train, min_loss)
            save_checkpoint({
                "epoch": epoch,
                "arch": "DeepSEA",
                "state_dict": self.model.state_dict(),
                "min_loss": min_loss,
                "optimizer": self.optimizer.state_dict()}, is_best)


class DeepSEA(nn.Module):
    def __init__(self, window_size, n_classes):
        super(DeepSEA, self).__init__()
        # features => the model's features.
        # not the same as genomic features, which we now refer to as classes,
        # per the convention in the deep learning model architectures.
        self.features = nn.Sequential(
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
            nn.Linear(960 * self.n_channels, n_classes),
            nn.ReLU(inplace=True),
            nn.Linear(n_classes, n_classes),

            nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 960 * self.n_channels)
        x = self.classifier(x)
        return x


def deepsea(filepath=None, **kwargs):
    model = DeepSEA(**kwargs)
    if filepath is not None:
        model.load_state_dict(torch.load(filepath))
        model.eval()
    return model
