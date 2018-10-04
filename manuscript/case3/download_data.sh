#!/bin/sh

############################################
# Data used for variant effect prediction
############################################

wget https://zenodo.org/record/1445556/files/variant_effect_prediction_data.tar.gz

tar -xzvf variant_effect_prediction_data.tar.gz -C ./1_variant_effect_prediction

##############################################################
# Data used for the variant groups comparison
##############################################################

wget https://zenodo.org/record/1445556/files/IGAP_lt0.05.npz

wget https://zenodo.org/record/1445556/files/IGAP_gt0.50.npz

###########################
# Outputs
###########################


# NOTE: outputs generated are commented out because the files are quite large.
# Please un-comment them if you would like to download the output directories.


# Output .tsv files from variant effect prediction

# wget https://zenodo.org/record/1445556/files/predict_outputs.tar.gz

# tar -xzvf predict_outputs.tar.gz

# Output .npz, .txt, .png files from variant groups comparison
# (generated Fig 3b.)

# wget https://zenodo.org/record/1445556/files/comparison_outputs.tar.gz

# tar -xzvf comparison_outputs.tar.gz
