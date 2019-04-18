"""
Description:
    TODO

Output:
    TODO

Usage:
    vep_cli.py <vcf> <reference-fasta> <output-dir>
    vep_cli.py -h | --help

Options:
    -h --help               Show this screen.

    <vcf>                   Input VCF file
    <reference-fasta>       Reference genome file
    <output-dir>            Output directory
"""
import os

from docopt import docopt

from selene_sdk.sequences import Genome
from selene_sdk.utils import load_path
from selene_sdk.utils import parse_configs_and_run
from selene_sdk import __version__


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version=__version__)

    def run_config(config_yml, output_dir):
        configs = load_path(config_yml, instantiate=False)
        reference_fa = Genome(arguments["<reference-fasta>"])
        configs["analyze_sequences"].bind(reference_sequence=reference_fa)
        configs["variant_effect_prediction"].update(
            vcf_files=[arguments["<vcf>"]],
            output_dir=output_dir)
        parse_configs_and_run(configs)

    deepsea_out = os.path.join(arguments["<output-dir>"], "deepsea")
    os.makedirs(deepsea_out, exist_ok=True)
    run_config("configs/expecto_deepsea.yml", deepsea_out)

