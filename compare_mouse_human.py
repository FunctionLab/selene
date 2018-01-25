import os

import torch

from deep_resnet import DeepSEA as DeepSEA_Res
from evaluate_model import evaluate
from sampler import ChromatinFeaturesSampler
from sampler_utils import sample_sequences


def get_features_list(features_file):
    features_list = None
    with open(features_file, 'r') as file_handle:
        lines = file_handle.readlines()
        features_list = [l.strip() for l in lines]
    return features_list


if __name__ == "__main__":
    #hg19_genome = "/tigress/kc31/hg19/male.hg19.fasta"
    mm10_genome = "/tigress/kc31/mm10/mm10_no_alt_analysis_set_ENCODE.fasta"

    """
    hg19_dir = "/tigress/kc31/compress_hg19_latest"
    hg19_features = os.path.join(hg19_dir, "distinct_features.txt")
    hg19_coords_only = os.path.join(hg19_dir, "coords_only.txt")
    hg19_query_tabix = os.path.join(hg19_dir, "sorted_aggregate.bed.gz")
    """

    mm10_dir = "/tigress/kc31/compress_mm10_latest"
    mm10_features = os.path.join(mm10_dir, "distinct_features.txt")
    mm10_coords_only = os.path.join(mm10_dir, "coords_only.txt")
    mm10_query_tabix = os.path.join(mm10_dir, "sorted_aggregate.bed.gz")

    test_chrs = ["chr8", "chr9"]

    n_samples = 6400
    batch_size = 16

    mm10_features_list = get_features_list(mm10_features)

    mm10_resnet = "/tigress/kc31/outputs_mm10/2018-01-16-23-18-35/best_model.pth.tar"
    """
    hg19_resnet = "/tigress/kc31/outputs/2018-01-17-16-05-41/best_model.pth.tar"
    hg19_origin = "/tigress/kc31/outputs/2018-01-17-10-58-17/best_model.pth.tar"
    """

    output_dir = "/home/kc31/fi-projects/deepsea-pipeline/data/labmtg_mm10"

    sampler = ChromatinFeaturesSampler(
        mm10_genome, mm10_query_tabix, mm10_coords_only, mm10_features,
        test_chrs, mode="test", sample_from="positive")

    sequence_encodings = sample_sequences(sampler, n_samples, output_dir)

    resnet_model = DeepSEA_Res(1001, len(mm10_features_list))
    load_model = torch.load(mm10_resnet)
    resnet_model.load_state_dict(load_model["state_dict"])
    resnet_model.cuda()
    resnet_model.eval()

    evaluate(resnet_model, batch_size, sequence_encodings, mm10_features_list,
             os.path.join(output_dir, "resnet_predictions.txt"))



