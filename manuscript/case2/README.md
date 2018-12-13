# Case 2: Developing a new model architecture and making model comparisons

For this case study, we do the following:

1. Train a new architecture on data from a deep learning publication (DeepSEA). We do this in 2 ways:
    - Using the exact `.mat` files released for the publication, see ([`1_train_with_deepsea_mats`](https://github.com/FunctionLab/selene/tree/master/manuscript/case2/1_train_with_deepsea_mats))
    - Using the chromatin profiles (`.bed` files) underlying the processed `.mat` files, see ([`1_train_with_online_sampler`](https://github.com/FunctionLab/selene/tree/master/manuscript/case2/1_train_with_online_sampler))

2. Evaluate new model using the DeepSEA test dataset ([`2_model_comparison`](https://github.com/FunctionLab/selene/tree/master/manuscript/case2/2_model_comparison))

The file `download_data.sh` will download all the data and outputs required to run each of these steps.
The rest of this README assumes that you have run this script.
Any directories mentioned in the README that are not included by default should have been downloaded using `download_data.sh`.
We have included comments in that file with more information about what is downloaded. 
You can view the files names and some descriptions at [this Zenodo record](https://doi.org/10.5281/zenodo.2214970) as well.

NOTE: The steps that we provide in this directory use input data that was processed from ENCODE and Roadmap Epigenomics.
Please consult the methods section in our [manuscript](https://doi.org/10.1101/438291) for a detailed summary of what data processing steps were taken.
The code to implement these steps can be viewed in `data/process_data.sh` after running `download_data.sh`.

## Step 1: train a new architecture

We provide example SLURM scripts and the YAML files we used to run training in `1_train_with_deepsea_mats` and `1_train_with_online_sampler`. Note that you need to modify the paths in the YAML files to use them. 

- The config in `1_train_with_deepsea_mats` uses Selene's [`selene_sdk.samplers.MultiFileSampler`](http://selene.flatironinstitute.org/samplers.html#multifilesampler), which takes as input file samplers for each of the training, validation, and testing datasets. 
- The config in `1_train_with_online_sampler` uses Selene's [`selene_sdk.samplers.IntervalsSampler`](http://selene.flatironinstitute.org/samplers.html#intervalssampler), which takes as input a tabix-indexed `.bed` file containing all the coordinates from the chromatin profiles, sorted and labeled according to the genomic feature/profile to which the row corresponds. In addition, it takes in a reference sequence (hg19 in this case, downloaded from ENCODE) FASTA, as well as a list of regions from which we should draw our samples. We restrict the regions to sample to the list of regions where there is at least one transcription factor present (in the chromatin profiles data from ENCODE/Roadmap Epigenomics).

For more information about these configurations, you can review the documentation page [here](https://selene.flatironinstitute.org/overview/cli.html).

The outputs from training and evaluation can be found in `mat_training_outputs` and `online_sampler_outputs`.

The `.sh` scripts in each of these directories runs Selene using [`../../../selene_cli.py`](https://github.com/FunctionLab/selene/blob/master/selene_cli.py) and a configuration file.
They also activate a conda environment called `selene-env`, the contents of which depend on how you would like to use Selene:

- The CLI script `../../../selene_cli.py` will try to run the local version of Selene (that is, this repository). It will work if you have built the Cython modules using `python setup.py build_ext --inplace`. In this case, your `selene-env` conda environment would not contain the `selene-sdk` Python package. Instead, it would contain all the dependencies of Selene (see: [`selene-gpu.yml`](https://github.com/FunctionLab/selene/blob/master/selene-gpu.yml)) as well as the `docopt`
  package (which parses the arguments for the CLI).
- If you want to use the installed `selene-sdk` package (through conda or pip), you can just move the CLI script to this case study's directory, update the `selene_cli.py` relative paths in the `.sh` scripts, and run the code for a specific case. (Make sure your `selene-env` contains `selene-sdk` and `docopt`.)

## Step 2: model comparison
_Working directory_: `2_model_comparison`

Please update the YAML file with the correct filepaths before using it.

We use Selene's [`selene_sdk.samplers.file_samplers.MatFileSampler`](http://selene.flatironinstitute.org/samplers.file_samplers.html#matfilesampler) to load DeepSEA's test dataset (`data/deepsea_test.mat`). The [`selene_sdk.EvaluateModel`](http://selene.flatironinstitute.org/selene.html#evaluatemodel) class is used to run this evaluation.  

The outputs from evaluating the model trained using the online sampler can be found in `evaluation_outputs`. 
