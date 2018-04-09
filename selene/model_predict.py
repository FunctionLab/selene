import numpy as np
import torch
from torch.autograde import Variable


def evaluate_test_sampler(model,
                          sampler,
                          batch_size,
                          features_list,
                          output_file):
    pass


def predict_on_encoded_sequences(model,
                                 sequences_mat,
                                 batch_size=64):
    predictions = []
    n_examples, _, _ = sequences_mat.shape
    for _ in range(0, n_examples, batch_size):
        predictions.append(model.forward(sequences_mat[n_examples, :, :]))
    return np.vstack(predictions)

def evaluate_test_sequences(model,
                            sequences_file,
                            batch_size,
                            features_list,
                            output_file):
    pass





