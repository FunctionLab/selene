![logo](docs/source/_static/img/selene_logo.png)

---

You have found *selene*, a Python library and command line interface for training deep neural networks from biological sequence data such as genomes.

## Installation

### Installing selene with pip:
```sh
pip install selene-sdk
```

### Installing selene with [Anaconda](https://www.anaconda.com/download/):
COMING SOON. Has not been uploaded to conda yet, though we will recommend using Selene with conda once it is up.
Currently, we recommend installing and running selene (using [selene-cli.py](selene-cli.py)) from the source build for better control over the PyTorch-related dependencies. 
```sh
conda install selene-sdk
```

### Installing selene from source:
```sh
git clone git@github.com:FunctionLab/selene
```
If you plan on working in the `selene` repository directly, we recommend setting up a conda environment using `selene-cpu.yml` or `selene-gpu.yml` and activating it. Please also build the Cython files by running
```sh
python setup.py build_ext --inplace
```

## Tutorials and examples

Tutorials for selene are available [here](https://github.com/FunctionLab/selene/tree/master/tutorials).

## Documentation

The documentation for selene is available [here](https://selene.flatironinstitute.org/).

