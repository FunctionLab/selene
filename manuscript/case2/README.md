# Case 2: Developing a new model architecture and making model comparisons

For this case study, we do the following:

1. Train a new architecture on data from a deep learning publication (DeepSEA). We do this in 2 ways:

- Using the exact `.mat` files released for the publication, see ([`1_train_with_deepsea_mats`](https://github.com/FunctionLab/selene/tree/master/manuscript/case2/1_train_with_deepsea_mats))
- Using the chromatin profiles (`.bed` files) underlying the processed `.mat` files, see ([`1_train_with_online_sampler`](https://github.com/FunctionLab/selene/tree/master/manuscript/case2/1_train_with_online_sampler))

2. Evaluate new model using the DeepSEA test dataset ([`2_model_comparison`](https://github.com/FunctionLab/selene/tree/master/manuscript/case2/2_model_comparison))

Finally, the file `download_data.sh` will download all the data and outputs required to run each of these steps.
The rest of this README assumes that you have run this script.
Any directories mentioned in the README that are not included by default should have been downloaded using `download_data.sh`.
We have included comments in that file with more information about what is downloaded. 

## Step 1: train a new architecture

Please refer to the [README in case1](https://github.com/FunctionLab/selene/tree/master/manuscript/case1/README.md) for general information about the SLURM scripts and YAML files in `1_train_with_deepsea_mats` and `1_train_with_online_sampler`. 

- The config in `1_train_with_deepsea_mats` uses Selene's [`selene_sdk.samplers.MultiFileSampler`](http://selene.flatironinstitute.org/samplers.html#multifilesampler), which takes as input file samplers for each of the training, validation, and testing datasets. 
- The config in `1_train_with_online_sampler` uses Selene's [`selene_sdk.samplers.IntervalsSampler`](http://selene.flatironinstitute.org/samplers.html#intervalssampler), which takes as input a tabix-indexed `.bed` file containing all the coordinates from the chromatin profiles, sorted and labeled according to the genomic feature/profile to which the row corresponds. In addition, it takes in a reference sequence (hg19 in this case, downloaded from ENCODE) FASTA, as well as a list of regions from which we should draw our samples. We restrict the regions to sample to the list of regions where there is at least one transcription factor present (in the chromatin profiles data from ENCODE/Roadmap Epigenomics).

The outputs from training and evaluation can be found in `mat_training_outputs` and `online_sampler_outputs`.

## Step 2: model comparison

The information in case1 about SLURM/YAML files applies to this step as well. 

We use Selene's [`selene_sdk.samplers.file_samplers.MatFileSampler`](http://selene.flatironinstitute.org/samplers.file_samplers.html#matfilesampler) to load DeepSEA's `test.mat` file (`1_train_with_deepsea_mats/data/test.mat`). The [`selene_sdk.EvaluateModel`](http://selene.flatironinstitute.org/selene.html#evaluatemodel) class is used to run this evaluation.  

The outputs from evaluating the model trained using the online sampler can be found in `evaluation_outputs`. 
