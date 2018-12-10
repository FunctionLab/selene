"""
Description:
    This script is used to create the file `TF_intervals.txt` from the file
    `sorted_deepsea_data.bed` that was created by running
    `process_chromatin_profiles.py`. `TF_intervals.txt` is used as input
    to a `selene_sdk.samplers.IntervalsSampler` and contains only the
    regions in `sorted_deepsea_data.bed` annotated to at least 1 TF.

Usage:
    create_TF_intervals_file.py <features> <data-bed> <output-txt>
    create_TF_intervals_file.py -h | --help

Options:
    -h --help            Show this screen
    <features>           Path to the list of genomic features in our dataset
    <data-bed>           Path to the DeepSEA data .bed file
    <output-txt>         Path to the output file

"""


from docopt import docopt


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")
    features_file = arguments["<features>"]
    data_file = arguments["<data-bed>"]
    output_file = arguments["<output-txt>"]

    features = []
    with open(features_file, 'r') as fh:
        for f in fh:
            features.append(f.strip())

    only_TF_features = []
    for f in features:
        if "DNase" in f:
            continue
        f_sep = f.split('|')
        target = f_sep[1]
        # if-statement to check for histone mark features
        if 'H' == target[0] and str.isdigit(target[1]):
            continue
        only_TF_features.append(f)

    with open(data_file, 'r') as read_fh, \
            open(output_file, 'w+') as write_fh:
        for line in read_fh:
            cols = line.strip().split('\t')
            if cols[-1] not in only_TF_features:
                continue
            write_fh.write("{0}\t{1}\t{2}\n".format(
                cols[0], cols[1], cols[2]))

