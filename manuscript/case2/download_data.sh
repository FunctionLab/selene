#!/bin/sh

############################################################
# Data used for training with the online intervals sampler
############################################################

wget https://zenodo.org/record/2214970/files/online_sampler_data.tar.gz

tar -xzvf online_sampler_data.tar.gz -C ./1_train_with_online_sampler

# Commented out:
# Chromatin profiles download. ENCODE/Roadmap Epigenomics .bed files used
# in DeepSEA (Zhou & Troyanskaya, 2015).

# wget https://zenodo.org/record/2214970/files/chromatin_profiles.tar.gz

# tar -xzvf chromatin_profiles.tar.gz -C ./1_train_with_online_sampler/data

##############################################################
# Data used for training with the matrix (.mat) file sampler
##############################################################

wget https://zenodo.org/record/2214970/files/DeepSEA_data.tar.gz

tar -xzvf DeepSEA_data.tar.gz -C ./1_train_with_deepsea_mats

mv ./1_train_with_deepsea_mats/DeepSEA_data ./1_train_with_deepsea_mats/data

#######################################################################
# Data used for evaluating the deeper architecture model on DeepSEA's
# test.mat samples
#######################################################################

wget https://zenodo.org/record/2214970/files/model_comparison_data.tar.gz

tar -xzvf model_comparison_data.tar.gz -C ./2_model_comparison

###########################
# Outputs generated from
###########################


# NOTE: outputs generated are commented out. Please un-comment them if you
# would like to download the output directories.

# In particular, the `online_sampler_outputs` is quite large because
# we save all the training, validation, and testing data as .bed files.


# Training using the DeepSEA .mat files

# wget https://zenodo.org/record/2214970/files/mat_training_outputs.tar.gz

# tar -xzvf mat_training_outputs.tar.gz

# Training using the chromatin profiles and the intervals sampler

# wget https://zenodo.org/record/2214970/files/online_sampler_outputs.tar.gz

# tar -xzvf online_sampler_outputs.tar.gz

# Evaluating the model (trained using the intervals sampler) on the
# DeepSEA test.mat file

# wget https://zenodo.org/record/2214970/files/evaluation_outputs.tar.gz

# tar -xzvf evaluation_outputs.tar.gz
