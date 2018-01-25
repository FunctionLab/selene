import os

import numpy as np


def get_feature_set_for_sample(sampler, indicator_features_vec):
    sample_features = []
    for index, indicator in enumerate(indicator_features_vec):
        if indicator > 0:
            feature = sampler.get_feature_from_index(index)
            sample_features.append(feature)
    return set(sample_features)

def get_feature_str_for_sample(sampler, indicator_features_vec):
    sample_features = []
    for index, indicator in enumerate(indicator_features_vec):
        if indicator > 0:
            feature = sampler.get_feature_from_index(index)
            sample_features.append(feature)
    return ' @ '.join(sample_features)

def sample_sequences(sampler, draw_n_samples, output_dir):
    input_sequences = open(
        os.path.join(output_dir, "input_sequences.fasta"), 'w+')
    answers = open(
        os.path.join(output_dir, "answers.txt"), 'w+')
    queries = open(
        os.path.join(output_dir, "input_sequences_full.txt"), 'w+')

    batch_sequence_encoding, batch_features_present = sampler.sample_positive(
        draw_n_samples)

    n_features_in_each_sample = []

    for i in range(draw_n_samples):
        sequence_encoding = batch_sequence_encoding[i, :, :]
        features_present = batch_features_present[i, :]

        n_features_present = len(
            get_feature_set_for_sample(sampler, features_present))
        n_features_in_each_sample.append(n_features_present)

        sequence = sampler.get_sequence_from_encoding(sequence_encoding)
        features_str = get_feature_str_for_sample(sampler, features_present)
        input_sequences.write(">{0}\n".format(features_str))
        input_sequences.write("{0}\n".format(sequence[:len(sequence) - 1]))
        answers.write("{0}\n".format(features_str))
        queries.write("{0}\n".format(sequence))
    input_sequences.close()
    answers.close()
    queries.close()

    print("Average number of features in each sample: {0}".format(
        np.average(n_features_in_each_sample)))

    return batch_sequence_encoding
