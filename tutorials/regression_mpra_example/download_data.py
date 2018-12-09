import io
import gzip
import os
import urllib
import tarfile

import pandas
import scipy.io
import selene_sdk.sequences


def run():
    target_column = "rl"
    local_file = "sample_et_al.tar"

    # Download the data.
    urllib.retrieve("https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE114002&format=file", local_file)
    with tarfile.open(local_file, "r") as archive:
        contents = archive.extractfile("GSM3130435_egfp_unmod_1.csv.gz").read()
        contents = gzip.decompress(contents).decode("utf-8")
    os.remove(local_file)

    # Format data.
    df = pandas.read_csv(io.StringIO(contents), sep=",", index_col=0)
    df = df[["utr", "total_reads", target_column]]
    df.sort_values("total_reads", inplace=True, ascending=False)
    df.reset_index(inplace=True, drop=True)

    # Split into train/validation/test.
    df = df.iloc[:280000]
    datasets = dict(test=df.iloc[:20000])
    df = df.iloc[20000:]
    df = df.sample(frac=1.)
    datasets["validate"] = df.iloc[:20000]
    datasets["train"] = df.iloc[20000:]
    x = dict.fromkeys(datasets.keys())
    y = dict.fromkeys(datasets.keys())

    # Construct features.
    for k in datasets.keys():
        x[k] = list()
        y[k] = list()
        for i in range(datasets[k].shape[0]):
            x[k].append(selene_sdk.sequences.Genome.sequence_to_encoding(datasets[k]["utr"].iloc[i]).T)
            y[k].append(datasets[k][target_column].iloc[i])
        x[k] = numpy.stack(x[k])
        y[k] = numpy.asarray(y[k]).reshape(-1, 1)

    # Scale w/ parameters from training data to prevent leakage.
    sdev = numpy.std(y["train"])
    mean = numpy.mean(y["train"])
    for k in datasets.keys():
        y[k] = (y[k] - mean) / sdev

    # Write data to file.
    for k in datasets.keys():
        scipy.io.savemat("{0}.mat".format(k), dict(x=x[k], y=y[k]))


if __name__ == "__main__":
    run()

