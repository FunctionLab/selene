# Tutorials

This directory contains all of the tutorials for selene.

The most thorough tutorial for getting started is in [`getting_started_with_selene`](https://github.com/FunctionLab/selene/tree/master/tutorials/getting_started_with_selene).
To get started on training a model very quickly, please see [`quickstart_training`](https://github.com/FunctionLab/selene/tree/master/tutorials/quickstart_training).

Additionally, we have two tutorials that show how to apply trained models. Selene provides methods to run variant effect prediction and _in silico_ mutagenesis, along with some visualization methods that we recommend running based on our Jupyter notebook tutorials.

- Comprehensive _in silico_ mutagenesis tutorial: [`analyzing_mutations_with_trained_models`](https://github.com/FunctionLab/selene/tree/master/tutorials/analyzing_mutations_with_trained_models)
- Tutorial with both the config file method and the non-config file method of running Selene. Also shows how to run variant effect prediction and visualize the difference scores. Contains an _in silico_ mutagenesis example with known regulatory mutations: [`variants_and_visualizations`](https://github.com/FunctionLab/selene/tree/master/tutorials/variants_and_visualizations)

We also have a tutorial demonstrating Selene's use to predict mean ribosomal load based on 5' UTR sequences: [`regression_mpra_example`](https://github.com/FunctionLab/selene/tree/master/tutorials/regression_mpra_example). This is a good follow-up tutorial to the Getting Started tutorial if you are interested in training a regression model using Selene. It also shows how to run Selene with another model architecture. 

## Additional note
The log statements printed in the training tutorials are from running the tutorials on a CUDA-enabled machine. Run times will be far longer (as described in the [repository README](https://github.com/FunctionLab/selene#tutorials)) if you are running them only on CPU.

## Contributing tutorials

The process for adding a tutorial to selene is as follows:

1. Create a subdirectory in the tutorials directory. The name of this subdirectory should be the name of the tutorial, formatted in snake-case.
2. Write the tutorial in an [ipython notebook](https://ipython.org/notebook.html) in the subdirectory.
3. Store all data for the tutorial in the subdirectory, and create a gzipped archive (i.e. a `*.tar.gz` file) with all the data required for the tutorial.
4. Create a `*.nblink` link file in the `docs/source/tutorials` directory. This file will serve as a link to the tutorial's notebook file. Instructions for formatting this file can be found [here](https://github.com/vidartf/nbsphinx-link).
5. Add an entry for the tutorial to the list of tutorials in `docs/source/tutorials/index.rst`.
6. Rerun `make html` from the `docs` directory.

