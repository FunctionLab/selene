![logo](docs/source/_static/img/selene_logo.png)

---

You have found Selene, a Python library and command line interface for training deep neural networks from biological sequence data such as genomes.

## Installation

Selene is a Python 3+ package. We recommend using it with Python 3.6 or above. 

### Installing selene with [Anaconda](https://www.anaconda.com/download/):
COMING SOON. We are working on getting Selene registered on Bioconda. 
For now, we recommend installing and running Selene (using [selene_cli.py](selene_cli.py)) from the source build for better control over the PyTorch-related dependencies. Please see the instructions below for doing so.

```sh
conda install -c bioconda selene-sdk
```

### Installing selene with pip:
```sh
pip install selene-sdk
```

### Installing selene from source:

First, download the latest commits from the source repository:
```
git clone https://github.com/FunctionLab/selene.git
```

The `setup.py` script requires NumPy. Please make sure you have this already installed.

If you plan on working in the `selene` repository directly, we recommend [setting up a conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using `selene-cpu.yml` or `selene-gpu.yml` (if CUDA is enabled on your machine) and activating it.

Selene contains some Cython files. You can build these by running
```sh
python setup.py build_ext --inplace
```

Otherwise, if you would like to locally install Selene, you can run
```sh
python setup.py install
```

Please install `docopt` before running the command-line script `selene_cli.py` provided in the repository.

## Tutorials and examples

Tutorials for selene are available [here](https://github.com/FunctionLab/selene/tree/master/tutorials).

## Documentation

The documentation for selene is available [here](https://selene.flatironinstitute.org/).

