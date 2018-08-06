# Installation

We recommend that users either clone and build the repository or install Selene through conda. This is because there is less control over PyTorch dependencies with pip.

## Installing with Anaconda

We are in the process of uploading Selene to [Bioconda](https://bioconda.github.io/), a channel for the [conda](https://www.anaconda.com/download/) package manager. 

It should be uploaded within the next week. If you'd like to be notified when it is available, please send me an e-mail at <kchen@flatironinstitute.org>. Thanks for your patience!

To install with conda, use the following command in your terminal:
```
conda install -c bioconda selene-sdk
```

## Installing with pip

Selene can be installed with [pip](https://pypi.org/project/pip/).
To install with pip, simply type the following commands into the terminal:
```
pip install selene-sdk
```

## Installing from source

Selene can also be installed from source.
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
