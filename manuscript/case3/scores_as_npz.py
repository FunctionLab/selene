"""
Description:
    Convert the file outputted by variant effect prediction to
    a .npz file

Usage:
    scores_as_npz.py <input-path> <output-path>
    scores_as_npz.py -h | --help

Options:
    -h --help           Show this screen.
    <input-path>        Path to the input .tsv file.
    <output-path>       Path for the output .npz file.
"""
from docopt import docopt
import numpy as np


def save_matrix_as_npz(filepath, save_to_path):
    scores_mat = []
    with open(filepath, 'r') as fh:
        fh.readline()
        for line in fh:
            cols = line.strip().split('\t')
            scores = [float(s) for s in cols[5:]]
            scores_mat.append(scores)
    scores_mat = np.array(scores_mat)
    np.savez_compressed(
        save_to_path, data=scores_mat)

if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")

    input_path = arguments["<input-path>"]
    output_path = arguments["<output-path>"]

    save_matrix_as_npz(input_path, output_path)
