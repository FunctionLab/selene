import os

import torch

from deep_resnet import DeepSEA as DeepSEA_Res
from deepsea import DeepSEA
from evaluate_model import evaluate
from sampler import ChromatinFeaturesSampler
from sampler_utils import sample_sequences


if __name__ == "__main__":
    genome = "/tigress/kc31/hg19/male.hg19.fasta"

    hg19_dir = "/tigress/kc31/compress_hg19_latest"
    features = os.path.join(hg19_dir, "distinct_features.txt")
    coords_only = os.path.join(hg19_dir, "coords_only.txt")
    query_tabix = os.path.join(hg19_dir, "sorted_aggregate.bed.gz")

    test_chrs = ["chr8", "chr9"]

    n_samples = 6400
    batch_size = 16

    features_list = None
    with open(features, 'r') as file_handle:
        lines = file_handle.readlines()
        features_list = [l.strip() for l in lines]

    resnet = "/tigress/kc31/outputs/2018-01-17-16-05-41/best_model.pth.tar"
    origin = "/tigress/kc31/outputs/2018-01-17-10-58-17/best_model.pth.tar"

    output_dir = "/home/kc31/fi-projects/deepsea-pipeline/data/labmtg"

    sampler = ChromatinFeaturesSampler(
        genome, query_tabix, coords_only, features,
        test_chrs, mode="test", sample_from="positive")

    sequence_encodings = sample_sequences(sampler, n_samples, output_dir)

    resnet_model = DeepSEA_Res(1001, len(features_list))
    load_model = torch.load(resnet)
    resnet_model.load_state_dict(load_model["state_dict"])
    resnet_model.cuda()
    resnet_model.eval()

    evaluate(resnet_model, batch_size, sequence_encodings, features_list,
             os.path.join(output_dir, "resnet_predictions.txt"))

    origin_model = DeepSEA(1001, len(features_list))
    load_model = torch.load(origin)
    origin_model.load_state_dict(load_model["state_dict"])
    origin_model.cuda()
    origin_model.eval()

    evaluate(origin_model, batch_size, sequence_encodings, features_list,
             os.path.join(output_dir, "origin_predictions.txt"))




