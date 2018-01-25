import os

import numpy as np
import torch

from deep_resnet import DeepSEA as DeepSEA_Res
from deepsea import DeepSEA
from evaluate_model import evaluate
from proteus.genome import _sequence_to_encoding

def get_features_list(features_file):
    features_list = None
    with open(features_file, 'r') as file_handle:
        lines = file_handle.readlines()
        features_list = [l.strip() for l in lines]
    return features_list


if __name__ == "__main__":
    hg19_origin = "/tigress/kc31/outputs/2018-01-22-22-02-48/best_model.pth.tar"
    #hg19_origin = "/tigress/kc31/outputs/2018-01-23-12-06-49/best_model.pth.tar"
    hg19_resnet = "/tigress/kc31/outputs/2018-01-22-21-58-06/best_model.pth.tar"

    output_dir = "/home/kc31/fi-projects/deepsea-pipeline/data/labmtg"

    human_samples = "/home/kc31/fi-projects/deepsea-pipeline/data/labmtg/input_sequences_full.txt"

    hg19_dir = "/tigress/kc31/compress_hg19_latest"
    features = os.path.join(hg19_dir, "distinct_features.txt")
    features_list = get_features_list(features)

    human_samples_list = None
    with open(human_samples, 'r') as fh:
        lines = fh.readlines()
        human_samples_list = [l.strip() for l in lines]

    sequence_encodings = [
        _sequence_to_encoding(s, {'A': 0, 'C': 1, 'G': 2, 'T': 3})
        for s in human_samples_list
    ]
    sequence_encodings = np.array(sequence_encodings)

    origin_model = DeepSEA(1001, len(features_list))
    load_model = torch.load(hg19_origin)
    origin_model.load_state_dict(load_model["state_dict"])
    origin_model.cuda()
    origin_model.eval()

    batch_size = 16
    evaluate(origin_model, batch_size, sequence_encodings, features_list,
             os.path.join(output_dir, "origin_predictions.txt"))

    resnet_model = DeepSEA_Res(1001, len(features_list))
    load_model = torch.load(hg19_resnet)
    resnet_model.load_state_dict(load_model["state_dict"])
    resnet_model.cuda()
    resnet_model.eval()

    batch_size = 16
    evaluate(resnet_model, batch_size, sequence_encodings, features_list,
             os.path.join(output_dir, "resnet_predictions.txt"))



