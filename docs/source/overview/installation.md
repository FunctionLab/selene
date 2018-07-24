# Installation

Selene can be installed with [pip](https://pypi.org/project/pip/).
To install with pip, simply type the following commands into the terminal:
```
pip install selene-sdk
```

## Installing with Anaconda

Selene can be installed with [conda](https://www.anaconda.com/download/) as well.
To install with conda, use the following command in your terminal:
```
conda install selene-sdk
```

## Installing from source

Selene can also be installed from source.
First, download the latest commits from the source repository:
```
git clone https://github.com/FunctionLab/selene
```

If you plan on working in the `selene` repository directly, we recommend setting up a conda environment using `selene-cpu.yml` or `selene-gpu.yml` and activating it. Please also build the Cython files by running
```sh
python setup.py build_ext --inplace
```
