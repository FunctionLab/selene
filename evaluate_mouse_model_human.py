import os

import numpy as np
import torch

from deepsea import DeepSEA
from deep_resnet import DeepSEA as DeepSEA_Res
from evaluate_model import evaluate
from proteus.genome import _sequence_to_encoding

def get_features_list(features_file):
    features_list = None
    with open(features_file, 'r') as file_handle:
        lines = file_handle.readlines()
        features_list = [l.strip() for l in lines]
    return features_list


if __name__ == "__main__":
    #mm10_resnet = "/tigress/kc31/outputs_mm10/2018-01-16-23-18-35/best_model.pth.tar"
    mm10_resnet = "/tigress/kc31/outputs_mm10/2018-01-22-22-14-31/best_model.pth.tar"
    mm10_convnet = "/tigress/kc31/outputs_mm10/2018-01-24-10-40-28/best_model.pth.tar"

    output_dir = "/home/kc31/fi-projects/deepsea-pipeline/data/labmtg_mm10/on_human"

    human_samples = "/home/kc31/fi-projects/deepsea-pipeline/data/labmtg/input_sequences_full.txt"

    mm10_dir = "/tigress/kc31/compress_mm10_latest"
    mm10_features = os.path.join(mm10_dir, "distinct_features.txt")
    mm10_features_list = get_features_list(mm10_features)

    human_samples_list = None
    with open(human_samples, 'r') as fh:
        lines = fh.readlines()
        human_samples_list = [l.strip() for l in lines]

    sequence_encodings = [
        _sequence_to_encoding(s, {'A': 0, 'C': 1, 'G': 2, 'T': 3})
        for s in human_samples_list
    ]
    sequence_encodings = np.array(sequence_encodings)

    resnet_model = DeepSEA_Res(1001, len(mm10_features_list))
    load_model = torch.load(mm10_resnet)
    resnet_model.load_state_dict(load_model["state_dict"])
    resnet_model.cuda()
    resnet_model.eval()

    batch_size = 16
    evaluate(resnet_model, batch_size, sequence_encodings, mm10_features_list,
             os.path.join(output_dir, "resnet_predictions.txt"))

    convnet_model = DeepSEA(1001, len(mm10_features_list))
    load_model = torch.load(mm10_convnet)
    convnet_model.load_state_dict(load_model["state_dict"])
    convnet_model.cuda()
    convnet_model.eval()

    batch_size = 16
    evaluate(convnet_model, batch_size, sequence_encodings, mm10_features_list,
             os.path.join(output_dir, "convnet_predictions.txt"))



