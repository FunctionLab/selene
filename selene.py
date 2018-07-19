"""
Description:
    This script builds the model and trains it using user-specified input data.

Output:
    Saves model to a user-specified output file.

Usage:
    selene.py <config-yml> [--lr=<lr>]
    selene.py -h | --help

Options:
    -h --help               Show this screen.

    <config-yml>            Model-specific parameters
    --lr=<lr>               If training, the optimizer's learning rate
                            [default: None]
"""
from docopt import docopt

from selene.utils import load_path
from selene.utils import parse_configs_and_run
from selene import __version__


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version=__version__)

    configs = load_path(arguments["<config-yml>"], instantiate=False)
    parse_configs_and_run(configs, lr=arguments["--lr"])
