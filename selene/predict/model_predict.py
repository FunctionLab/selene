import itertools

import numpy as np
import torch
from torch.autograd import Variable

from .predict_handlers import DiffScoreHandler, LogitScoreHandler, \
        WritePredictionsHandler
from ..sequences import Genome
from ..sequences import sequence_to_encoding


def predict(model, batch_sequences, use_cuda=False):
    inputs = torch.Tensor(batch_sequences)
    if use_cuda:
        inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = model.forward(inputs.transpose(1, 2))
    return outputs


def predict_on_encoded_sequences(model,
                                 sequences,
                                 batch_size=64,
                                 use_cuda=False):
    predictions = []
    n_examples, _, _ = sequences.shape

    for i in range(0, n_examples, batch_size):
        start = i
        end = i + batch_size
        batch_sequences = sequences[start:end, :, :]
        outputs = predict(model, batch_sequences, use_cuda=use_cuda)
        predictions.append(outputs.data.cpu().numpy())
    return np.vstack(predictions)


def in_silico_mutagenesis_sequences(input_sequence,
                                    mutate_n_bases=1):
    """Creates a list containing each mutation that occurs from in silico
    mutagenesis across the whole sequence.

    Parameters
    ----------
    input_sequence : str
    mutate_n_bases : int

    Returns
    -------
    list
        A list of all possible mutations. Each element in the list is
        itself a list of tuples, e.g. [(0, 'T')] if we are only mutating
        1 base at a time. Each tuple is the position to mutate and the base
        with which we are replacing the reference base.

        For a sequence of length 1000, mutating 1 base at a time means that
        we return a list of length 3000.
    """
    sequence_alts = []
    for index, ref in enumerate(input_sequence):
        alts = []
        for base in Genome.BASES_ARR:
            if base == ref:
                continue
            alts.append(base)
        sequence_alts.append(alts)

    all_mutated_sequences = []
    for indices in itertools.combinations(
            range(len(input_sequence)), mutate_n_bases):
        pos_mutations = []
        for i in indices:
            pos_mutations.append(sequence_alts[i])
        for mutations in itertools.product(*pos_mutations):
            all_mutated_sequences.append(list(zip(indices, mutations)))
    return all_mutated_sequences


def _ism_sample_id(dna_sequence, mutation_information):
    positions = []
    refs = []
    alts = []
    for (position, alt) in mutation_information:
        positions.append(str(position))
        refs.append(dna_sequence[position])
        alts.append(alt)
    return (';'.join(positions), ';'.join(refs), ';'.join(alts))


def in_silico_mutagenesis_predict(model,
                                  batch_size,
                                  sequence,
                                  mutations_list,
                                  use_cuda=False,
                                  reporters=[]):
    current_sequence_encoding = sequence_to_encoding(
        sequence, Genome.BASE_TO_INDEX)
    for i in range(0, len(mutations_list), batch_size):
        start = i
        end = i + batch_size

        mutated_sequences = np.zeros(
            (batch_size, *current_sequence_encoding.shape))

        batch_ids = []
        for ix, mutation_info in enumerate(mutations_list[start:end]):
            mutated_seq = mutate_sequence(
                current_sequence_encoding, mutation_info)
            mutated_sequences[ix, :, :] = mutated_seq
            batch_ids.append(_ism_sample_id(sequence, mutation_info))
        outputs = predict(
            model, mutated_sequences, use_cuda=use_cuda).data.cpu().numpy()

        for r in reporters:
            r.handle_batch_predictions(outputs, batch_ids)

    for r in reporters:
        r.write_to_file()


def _reverse_strand(dna_sequence):
    reverse_bases = [Genome.COMPLEMENTARY_BASE[b] for b in dna_sequence[::-1]]
    return ''.join(reverse_bases)


def mutate_sequence(dna_encoded_sequence, mutation_information):
    mutated_seq = np.copy(dna_encoded_sequence)
    for (position, alt) in mutation_information:
        replace_base = Genome.BASE_TO_INDEX[alt]
        mutated_seq[position, :] = 0
        mutated_seq[position, replace_base] = 1
    return mutated_seq


def in_silico_mutagenesis(model,
                          batch_size,
                          input_sequence,
                          features_list,
                          save_diffs,
                          mutate_n_bases=1,
                          use_cuda=False,
                          save_logits=None,
                          save_predictions=None):
    mutated_sequences = in_silico_mutagenesis_sequences(
        input_sequence, mutate_n_bases=1)

    current_sequence_encoding = sequence_to_encoding(
        input_sequence, Genome.BASE_TO_INDEX)

    base_encoding = current_sequence_encoding.reshape(
        (1, *current_sequence_encoding.shape))
    base_preds = predict(
        model, base_encoding).data.cpu().numpy()

    reporters = []
    nonfeature_cols = ["pos", "ref", "alt"]
    if save_diffs:
        diff_handler = DiffScoreHandler(
            base_preds, features_list, nonfeature_cols, save_diffs)
        reporters.append(diff_handler)
    if save_logits:
        logit_handler = LogitScoreHandler(
            base_preds, features_list, nonfeature_cols, save_logits)
        reporters.append(logit_handler)
    if save_predictions:
        preds_handler = WritePredictionsHandler(
            features_list, nonfeature_cols, save_predictions)
        reporters.append(preds_handler)

    in_silico_mutagenesis_predict(
        model, batch_size, input_sequence, mutated_sequences,
        use_cuda=use_cuda, reporters=reporters)
