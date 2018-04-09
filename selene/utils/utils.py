import sys

import yaml


def read_yaml_file(config_file):
    with open(config_file, "r") as config_file:
        try:
            config_dict = yaml.load(config_file)
            return config_dict
        except yaml.YAMLError as exception:
            sys.exit(exception)
