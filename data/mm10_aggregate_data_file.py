"""
Usage:
    mm10_aggregate_data_file.py <input-dir> <bigBed-dir> <output-file>
    mm10_aggregate_data_file.py -h | --help

Options:
    -h --help        Show this screen.

    <input-dir>      Input directory. Must have a `metadata.tsv` file
                     that describes the rest of the ENCODE data files in
                     the directory.
    <bigBed-dir>     Directory to the uncompressed bigBed files that were
                     part of our ENCODE download. (Used bigBedToBed binary.)
    <output-file>    The output file. Concatenate all ENCODE data files,
                     where each file has been modified to include the genomic
                     feature detected in the assay as a new column, along with
                     the metadata index (the row number allows us to map the
                     data back to the file name).
"""
import os
from time import time

from docopt import docopt
import pandas as pd

if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")

    data_dir = arguments["<input-dir>"]
    bigBed_dir = arguments["<bigBed-dir>"]
    aggregate_file = arguments["<output-file>"]

    metadata_df = pd.read_table(os.path.join(data_dir, "metadata.tsv"))

    t_i = time()
    all_files = []
    n_positive_examples = 0

    for index, row in metadata_df.iterrows():
        filename = row["File download URL"].split("/")[-1]
        filepath = None
        if "bigBed" in filename:
            filepath = os.path.join(bigBed_dir, "{0}.bed".format(filename))
        else:
            filepath = os.path.join(data_dir, filename)

        try:
            bedfile_df = pd.read_table(
                filepath,
                usecols=[0, 1, 2, 5],
                names=["chr", "start", "end", "strand"],
                header=None)
            n_rows = len(bedfile_df)
            n_positive_examples += n_rows

            feature = []
            assay = row["Assay"].split("-")[0]
            biosample_term = row["Biosample term name"]
            if str(row["Experiment target"]) != "nan":
                target = row["Experiment target"].split("-")[0]
                feature = [assay, target, biosample_term]
            else:
                feature = [assay, " ", biosample_term]
            feature = "; ".join(feature)
            bedfile_df = bedfile_df.assign(
                feature=pd.Series([feature] * n_rows))
            bedfile_df = bedfile_df.assign(
                metadata_index=pd.Series([index] * n_rows))
            all_files.append(bedfile_df)
        except IOError:
            # Manually check these files - they should be files that
            # could not be successfully downloaded from ENCODE.
            print(filepath)
    aggregate_bed_df = pd.concat(all_files, ignore_index=True)
    print("Total positive examples over {0} files: {1}".format(
        index + 1, n_positive_examples))

    t_f = time()
    print("Time taken to aggregate all files: {0} s".format(
        t_f - t_i))

    aggregate_bed_df.to_csv(
        aggregate_file, sep="\t", header=False, index=False)
