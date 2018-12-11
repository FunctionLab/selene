"""
Description:
    Used for the _in silico_ mutagenesis section in this case study.
    Randomly selects some sequences from the test set to use.

Output:
    Saves randomly selected sequences to a FASTA file.

Usage:
    get_test_regions.py <features-bed> <genome-fa> <output-fa>
                        [--seq-len=<len>] [--n-samples=<N>]
                        [--holdouts=<holdouts>] [--seed=<seed>]
    get_test_regions.py -h | --help

Options:
    -h --help             Show this screen

    <features-bed>        Features .bed filepath
    <genome-fa>           Genome FASTA filepath
    <output-fa>           Output FASTA filepath

    --seq-len=len         Sequence length
                          [default: 1000]
    --n-samples=N         Number of random sequences to write to file
                          [default: 20]
    --holdouts=holdouts   Chromosomal holdouts (note this script was
                          not written to support proportional holdouts).
                          Multiple chromosomes should be separated by
                          a comma ','.
                          [default: chr8,chr9]
    --seed=seed           Random seed
                          [default: 42]
"""
from docopt import docopt
import numpy as np
from pyfaidx import Fasta


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="0.0.0")

    features_bedfile = arguments["<features-bed>"]
    genome_fastafile = arguments["<genome-fa>"]
    output_fastafile = arguments["<output-fa>"]
    sequence_length = int(arguments["--seq-len"])
    n_samples = int(arguments["--n-samples"])
    rseed = int(arguments["--seed"])

    genome = Fasta(genome_fastafile)
    chromosomes = arguments["--holdouts"].split(',')
    coordinates = []
    with open(features_bedfile, 'r') as file_handle:
        for line in file_handle:
            cols = line.strip().split('\t')
            if cols[0] not in chromosomes:
                continue
            start = int(cols[1])
            end = int(cols[2])
            coordinates.append((cols[0], start, end))
    np.random.seed(rseed)
    selected_ixs = np.random.choice(
        range(len(coordinates)), n_samples, replace=False)
    selected_coordinates = []
    for ix in selected_ixs:
        selected_coordinates.append(coordinates[ix])

    with open(output_fastafile, 'w+') as file_handle:
        for (chrom, start, end) in selected_coordinates:
            region_len = end - start
            midpoint = start + region_len // 2
            seq_start = midpoint - np.floor(sequence_length / 2.)
            seq_end = midpoint + np.ceil(sequence_length / 2.)

            if seq_start < 0 or seq_start > len(genome[chrom]):
                continue

            sequence = genome[chrom][int(seq_start):int(seq_end)].seq
            file_handle.write(">{0}_{1}_{2}\n".format(chrom, start, end))
            file_handle.write("{0}\n".format(sequence))

