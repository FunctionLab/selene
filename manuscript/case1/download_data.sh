#!/bin/sh

# All the data files needed for both training/evaluation and _in silico_
# mutagenesis.

wget https://zenodo.org/record/2214130/files/data.tar.gz

tar -xzvf data.tar.gz

# Outputs from training (e.g. log files, test performance, figures)

wget https://zenodo.org/record/2214130/files/training_outputs.tar.gz

tar -xzvf training_outputs.tar.gz

# Outputs from _in silico_ mutagenesis (e.g. logit scores)

wget https://zenodo.org/record/2214130/files/ism_outputs.tar.gz

tar -xzvf ism_outputs.tar.gz

