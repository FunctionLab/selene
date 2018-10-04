# Selene: a PyTorch-based deep learning library for sequence-level data

Here we provide all the code needed to reproduce the case studies described in the Selene manuscript.

Please consult the READMEs for each case that you are interested in before trying to run any of the code.
Many of the data files are in Zenodo records and must be downloaded via a `download_data.sh` script beforehand. 
We have included scripts in the Zenodo records that show how to reproduce the data processing that we've done.

## Environment

We recommend installing `selene-sdk` with Anaconda or building the local version of Selene before running these examples.
You will need access to GPU nodes (we provide some example SLURM scripts that you can modify if your institution also uses SLURM) to run these cases in a reasonable amount of time.

If you use the CLI file we provide in this repository ([`selene_cli.py`](https://github.com/FunctionLab/selene/blob/master/selene_cli.py)) you should also add `docopt` to your conda environment.
Note that the CLI, in its current location in the repository, will try to use the local version of Selene. 
If you are conda installing Selene, please move the CLI file and adjust the paths in the SLURM scripts for each of the case studies.

If you are interested in running any of the data processing scripts, make sure that you have `tabix` and `bgzip` on your machine. You can either conda-install `htslib` to your environment or download and build it using [these instructions](http://www.htslib.org/download/).

Otherwise, you should be able to run all cases with either of these 2 setups:

- `conda install selene-sdk -c bioconda ; conda install docopt -c anaconda`
- `conda env create -f selene-gpu.yml ; conda install docopt -c anaconda`


