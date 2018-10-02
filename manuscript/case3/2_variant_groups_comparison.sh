#!/bin/bash

# Save the scores in `predict_outputs/*.tsv` files as compressed `*.npz`
# NumPy matrices
python scores_as_npz.py ./predict_outputs_1/lt0.05_igap_abs_diffs.tsv \
                        ./IGAP_lt0.05.npz

python scores_as_npz.py ./predict_outputs_1/gt0.50_igap_abs_diffs.tsv \
                        ./IGAP_gt0.50.npz

# For each genomic feature, compare variant groups (nominally significant and
# nonsignificant) using the 1-sided Wilcoxon rank sum test.
# This script generates our Figure 3b in the manuscript.
python variant_groups_comparison.py \
           ./1_variant_effect_prediction/data/distinct_features.txt \
           ./IGAP_gt0.50.npz \
           ./IGAP_lt0.05.npz \
           ./comparison_outputs
