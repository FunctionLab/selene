import logging
import sys

import yaml


VCF_REQUIRED_COLS = ["#CHROM", "POS", "ID", "REF", "ALT"]


def initialize_logger(out_filepath, verbosity=1, stdout_handler=False):
    """This function can only be called successfully once.
    If the logger has already been initialized with handlers,
    the function exits. Otherwise, it proceeds to set the
    logger configurations.
    """
    logger = logging.getLogger("selene")
    # check if logger has already been initialized
    if len(logger.handlers):
        return

    if verbosity == 0:
        logger.setLevel(logging.WARN)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    elif verbosity == 2:
        logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handle = logging.FileHandler(out_filepath)
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)

    if stdout_handler:
        stream_handle = logging.StreamHandler(sys.stdout)
        stream_handle.setFormatter(formatter)
        logger.addHandler(stream_handle)


def read_yaml_file(config_file):
    with open(config_file, "r") as config_file:
        try:
            config_dict = yaml.load(config_file)
            return config_dict
        except yaml.YAMLError as exception:
            sys.exit(exception)


def read_vcf_file(vcf_file):
    """Read the relevant columns for a VCF file to collect variants
    for variant effect prediction.
    """
    variants = []
    with open(vcf_file, 'r') as file_handle:
        for line in file_handle:
            if "#CHROM" in line:
                cols = line.strip().split('\t')
                if cols[:5] != VCF_REQUIRED_COLS:
                    raise ValueError(
                        "First 5 columns in file {0} were {1}. "
                        "Expected columns: {2}".format(
                            vcf_file, cols[:5], VCF_REQUIRED_COLS))
                break

        for line in file_handle:
            cols = line.strip().split('\t')
            chrom = str(cols[0])
            pos = int(cols[1])
            ref = cols[3]
            alt = cols[4]
            variants.append((chrom, pos, ref, alt))
    return variants

