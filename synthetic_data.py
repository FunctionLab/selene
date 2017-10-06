import numpy as np
import torch
from torch.autograd import Variable

from deepsea import DeepSEA
#from simple_model import DeepSEA

# NOTE: I've just been commenting in/out the deepsea.py
# vs. simple_model.py (linear model) scripts when testing

BASES = np.array(['A', 'T', 'G', 'C'])
N, D_in, H, D_out = 16, 4, 1001, 1


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

    xs = xs.cuda()
    ys = ys.cuda(async=True)
    return (Variable(xs), Variable(ys))

N_train = 96
N_validate = 32
batch_size = 128

print("Train: {0}, validate: {1}, batch size: {2}".format(N_train, N_validate, batch_size))

if __name__ == "__main__":
    # generate fixed datasets for training and validation
    train = []
    for _ in range(N_train):
        train.append(
            generate_data_bases(batch_size))

    validation = []
    for _ in range(N_validate):
        validation.append(
            generate_data_bases(batch_size))

    learning_rate = 1e-4
    print(learning_rate)

    model = DeepSEA(H, D_out)
    print(model)



    criterion = torch.nn.BCEWithLogitsLoss()


    # NOTE: I only started putting in the weight decay parameter,
    # where the value is the same as what you report in DeepSEA paper,
    # after I confirmed that the model could overfit the synthetic training
    # dataset

    #optimizer = torch.optim.SGD(model.parameters(),
    #    lr=learning_rate, momentum=0.95, weight_decay=5e-7)
    optimizer = torch.optim.Adam(model.parameters(),
        lr=learning_rate, weight_decay=5e-7)
    print(optimizer)

    model.cuda()
    criterion.cuda()

    for t in range(2000):
        model.train()


        # NOTE: This was code used when I wanted to
        # keep some percentage of the training dataset fixed
        # but also add some new examples each epoch.
        # Code takes much longer to run when this is executed.

        #for _ in range(int(N_train / 2)):
        #    train.append(
        #        generate_data_bases(batch_size))

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
