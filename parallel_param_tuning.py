import os
import sys
from time import time

from joblib import Parallel, delayed
import numpy as np
import torch
from torch.autograd import Variable

from deepsea import DeepSEA
#from simple_model import DeepSEA

"""This code has not been debugged yet. It runs but
has problems redirecting output to stdout, so it's pretty
useless right now. :( Feel free to modify if it is helpful
and let me know if you have made it into something nice!

Or tell me you're interested in this and I can make it work haha.
"""

BASES = np.array(['A', 'T', 'G', 'C'])
N, D_in, H, D_out = 16, 4, 1001, 1

N_train = 32
N_validate = 8
batch_size = 128

print("Train: {0}, validate: {1}, batch size: {2}".format(N_train, N_validate, batch_size))


def sequence_encoding(sequence):
     """Converts an input sequence to its one hot encoding.

     Parameters
     ----------
     sequence : str
         The input sequence of length N.

     Returns
     -------
     numpy.ndarray, dtype=bool
         The N-by-4 encoding of the sequence.
     """
     encoding = np.zeros((len(sequence), 4), np.bool_)
     for base, index in zip(sequence, range(len(sequence))):
         encoding[index, :] = BASES == base
     return encoding


def generate_random_sequence(n_examples):
    return np.random.choice(BASES, size=H * n_examples, replace=True)


def get_sequence_feature(x):
    num_gc = np.count_nonzero(x == 'G') + np.count_nonzero(x == 'C')
    if num_gc / float(len(x)) >= 0.50:
        return 1
    else:
        return 0


def generate_data_bases(N):
    xs = []
    ys = []
    seqs = generate_random_sequence(N)
    for i in range(N):
        x = seqs[i * H:(i + 1) * H]
        xs.append(sequence_encoding(x).T)
        ys.append(get_sequence_feature(x))

    xs = np.array(xs) * 1
    xs = torch.from_numpy(xs)
    xs = xs.float()
    #xs = xs.view(N, D_in, H, 1)

    ys = torch.from_numpy(np.array(ys))
    ys = ys.view(N, D_out)
    ys = ys.float()

    #xs = xs.cuda()
    #ys = ys.cuda(async=True)
    return (Variable(xs), Variable(ys))


def train_model(lr, optim):
    sys.stdout = open(str(os.getpid()) + ".out", "w")
    print("Learning rate: {0}".format(lr))
    model = DeepSEA(H, D_out)
    print(model)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = None
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
            lr=lr, momentum=0.95, weight_decay=5e-7)
    elif optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
            lr=lr, weight_decay=5e-7)
    elif optim == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(),
            lr=lr)
    print(optimizer)

    #model.cuda()
    #criterion.cuda()

    for t in range(6000):
        model.train()

        avg_loss_train = 0.
        for x_, y_ in train:
            y_pred = model(x_)
            # Compute and print loss
            loss = criterion(y_pred, y_)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss_train += loss.data[0]
        avg_loss_train /= len(train)

        model.eval()

        avg_loss_validate = 0.
        for xv_, yv_ in validation:
            yv_pred = model(xv_)
            lossv = criterion(yv_pred, yv_)
            avg_loss_validate += lossv.data[0]
        avg_loss_validate /= len(validation)

        print(t, avg_loss_train, avg_loss_validate)


if __name__ == "__main__":
    n_cores = int(sys.argv[1])
    optim = sys.argv[2]

    validation = []
    for _ in range(N_validate):
        validation.append(
            generate_data_bases(batch_size))

    train = []
    for _ in range(N_train):
        train.append(
            generate_data_bases(batch_size))

    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 0.1]

    t_i = time()
    with Parallel(n_jobs=n_cores) as parallel:
        results = parallel(
            delayed(train_model)(lr, optim) for lr in learning_rates)
    t_f = time()

    print("Finished parameter tuning in {0} seconds, {1} parameters tested".format(
        t_f - t_i, len(learning_rates)))
