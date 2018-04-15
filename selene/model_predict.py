import itertools

import numpy as np
import torch
from torch.autograd import Variable

from .sequences import Genome, sequence_to_encoding


def mat_predict_on_encoded_sequences(model,
                                     sequences_mat,
                                     batch_size=64,
                                     use_cuda=False):
    predictions = []
    n_examples, _, _ = sequences_mat.shape
    for i in range(0, n_examples, batch_size):
        start = i
        end = i + batch_size
        batch_sequences = sequences_mat[start:end, :, :]

        inputs = torch.Tensor(batch_sequences)
        if use_cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs = model.forward(inputs.transpose(1, 2))
        predictions.append(outputs.data.cpu().numpy())
    return np.vstack(predictions)


def in_silico_mutagenesis_sequences(input_sequence,
                                    mutate_n_bases=1,
                                    save_to_file=None):
    sequence_alts = []
    for index, base in enumerate(input_sequence):
        alts = []
        for other_base in Genome.BASES_ARR:
            if base == other_base:
                continue
            alts.append(other_base)
        sequence_alts.append(alts)

    all_mutated_sequences = []
    for indices in itertools.combinations(
            range(len(input_sequence)), mutate_n_bases):
        print(indices)
        pos_mutations = []
        for i in indices:
            pos_mutations.append(sequence_alts[i])
        for mutations in itertools.product(*pos_mutations):
            all_mutated_sequences.append(list(zip(indices, mutations)))
    print(len(all_mutated_sequences))
    print(all_mutated_sequences[:10])
    return all_mutated_sequences

def predict(model, batch_sequences, use_cuda=False):
    inputs = torch.Tensor(batch_sequences)
    if use_cuda:
        inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = model.forward(inputs.transpose(1, 2))
    return outputs

def in_silico_mutagenesis_predict(model,
                                  batch_size,
                                  original_sequence,
                                  mutations_list,
                                  save_to_file,
                                  use_cuda=False):
    sequence_encoding = sequence_to_encoding(original_sequence)
    # dim = [1, len(original_sequence), 4]
    batch_sequence_encoding = sequence_encoding.reshape(
        (1, *sequence_encoding.shape))
    base_prediction = predict(
        model, batch_sequence_encoding).data.cpu().numpy()

    file_handle = open(save_to_file, 'w+')
    for i in range(0, len(mutations_list), batch_size):
        start = i
        end = i + batch_size
        batch_sequences = np.zeros((batch_size, len(original_sequence), 4))
        for ix, mutation_info in enumerate(mutations_list[start:end]):
            ref_alt_list = []
            mutated_seq = np.copy(sequence_encoding)
            for (sequence_pos, alt) in mutation_info:
                ref = original_sequence[sequence_pos]
                replace_base = Genome.BASES_DICT[alt]
                mutated_seq[sequence_pos, :] = 0
                mutated_seq[sequence_pos, replace_base] = 1
                ref_alt_list.append(
                    "{0}-{1}-{2}".format(sequence_pos, ref, alt))
            batch_sequences[ix, :, :] = mutated_seq
        outputs = predict(model, batch_sequences)
        predictions = outputs.data.cpu().numpy()

        str_predictions = ["{:.2e}".format(p) for p in predictions]
        mutations_col = ';'.join(ref_alt_list)
        predict_cols = '\t'.join(str_predictions)
        file_handle.write("{0}\t{1}\n".format(mutations_col, predict_cols))
    file_handle.close()


def in_silico_mutagenesis(model,
                          batch_size,
                          input_sequence,
                          save_diffs_to_file,
                          mutate_n_bases=1,
                          save_logit_to_file=None,
                          save_predictions_to_file=None):
    raise NotImplementedError
