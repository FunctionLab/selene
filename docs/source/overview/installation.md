# Installation

Users can either clone and build the repository locally or install Selene through conda. We previously supported installation through pip, but are refraining from releasing the latest version of Selene through pip due to some issues we are observing when using the pip-installed torch and torchvision dependencies.

## Installing with Anaconda

To install with conda (recommended for Linux users), run the following command in your terminal:
```
conda install -c bioconda selene-sdk
```

## Installing from source

Selene can also be installed from source.
First, download the latest commits from the source repository:
```
git clone https://github.com/FunctionLab/selene.git
```

The `setup.py` script requires NumPy, Cython, and setuptools. Please make sure you have these already installed.

If you plan on working in the `selene` repository directly, we recommend [setting up a conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using `selene-cpu.yml` or `selene-gpu.yml` (if CUDA is enabled on your machine) and activating it.

Selene contains some Cython files. You can build these by running
```sh
python setup.py build_ext --inplace
```

If you would like to locally install Selene, you can run
```sh
python setup.py install
```

## Additional dependency for running the CLI 

Please install `docopt` before running the command-line script `selene_cli.py` provided in the repository.
