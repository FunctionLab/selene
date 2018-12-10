"""
Description:
    This script is used to compile the DeepSEA dataset of 919 genomic
    features.

Usage:
    process_chromatin_profiles.py <features>
                            <ENCODE-DNase> <ENCODE-TF> <Roadmap>
                            <output-dir>
    process_chromatin_profiles.py -h | --help

Options:
    -h --help               Show this screen.

    <features>              Path to the list of features the DeepSEA model
                            predicts.
    <ENCODE-DNase>          Path to the directory with ENCODE DNase files
                            and metadata file.
    <ENCODE-TF>             Path to the directory with ENCODE TF files and
                            metadata file.
    <Roadmap>               Path to the directory with Roadmap Epigenomics
                            files and metadata file.
    <output-dir>            Path to the desired output directory. If it does
                            not exist, it will automatically be created for
                            you.
"""
import os

from docopt import docopt
import pandas as pd


def _format_feature_dataset(filepath, feature):
    file_df = pd.read_table(filepath, header=None, usecols=[0, 1, 2, 3])
    n_rows, _ = file_df.shape
    file_df = file_df.assign(feature=pd.Series([feature] * n_rows))
    file_df = file_df.assign(metadata_index=pd.Series([index] * n_rows))
    return file_df


def metadata_to_dict(filepath):
    metadata = {}
    with open(filepath, "r") as fh:
        for line in fh:
            dataset, ds_info = line.split("\t")
            info_dict = {}
            key_vals = ds_info.split(";")
            for kv in key_vals:
                kv = kv.strip()
                key, val = kv.split("=")
                info_dict[key] = val
            metadata[dataset] = info_dict
    return metadata


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")
    features_file = arguments["<features>"]
    ENCODE_DNase = arguments["<ENCODE-DNase>"]
    ENCODE_TF = arguments["<ENCODE-TF>"]
    Roadmap_Epi = arguments["<Roadmap>"]

    ENC_DNase_file = os.path.join(
        ENCODE_DNase, "wgEncodeAwgDnase")
    ENC_TF_file = os.path.join(
        ENCODE_TF, "wgEncodeAwgTfbs")
    Roadmap_file = os.path.join(
        Roadmap_Epi,
        "jul2013.roadmapData.qc - "
        "Consolidated_EpigenomeIDs_summary_Table.tsv")

    output_dir = arguments["<output-dir>"]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    DNase_metadata = metadata_to_dict(ENC_DNase_file)
    TFs_metadata = metadata_to_dict(ENC_TF_file)
    Roadmap_metadata = pd.read_table(Roadmap_file)
    Roadmap_metadata.set_index("Epigenome ID (EID)", inplace=True)

    # unique features mapped to number of duplicates
    deepsea_features = {}
    with open(features_file, "r") as file_handle:
        for line in file_handle:
            feature = line.strip()
            if feature not in deepsea_features:
                deepsea_features[feature] = 0
            deepsea_features[feature] += 1

    all_features_with_dups = []
    all_features = []
    dfs_to_concat = []

    # ENCODE DNase features
    for index, (filename, info) in enumerate(DNase_metadata.items()):
        # just to handle some edge cases
        filename_split = filename.split(".")
        if len(filename_split) > 3:
            filename = "".join(filename_split[:-2]) + ".narrowPeak.gz"
        filepath = os.path.join(ENCODE_DNase, filename)
        feature = "{0}|DNase|{1}".format(info["cell"], info["treatment"])
        if feature not in deepsea_features:
            continue
        all_features_with_dups.append(feature)

        if feature in deepsea_features and deepsea_features[feature] > 1:
            deepsea_features[feature] -= 1
            feature = "{0}|{1}".format(feature, deepsea_features[feature])
        elif feature in deepsea_features and deepsea_features[feature] == 1:
            deepsea_features[feature] -= 1
        elif feature in deepsea_features and deepsea_features[feature] <= 0:
            continue
        all_features.append(feature)
        dfs_to_concat.append(_format_feature_dataset(
            filepath, feature))

    DNase_agg = pd.concat(dfs_to_concat, ignore_index=True)
    DNase_agg.sort_values([0, 1, 2], ascending=True, inplace=True)

    print(DNase_agg.head())

    dfs_to_concat = []

    # ENCODE TF features
    for index, (filename, info) in enumerate(TFs_metadata.items()):
        filepath = os.path.join(ENCODE_TF, filename)
        if not os.path.isfile(filepath):
            continue
        feature = "{0}|{1}|{2}".format(
            info["cell"], info["antibody"].split('_')[0], info["treatment"])
        if feature not in deepsea_features:
            continue
        all_features_with_dups.append(feature)
        if feature in deepsea_features and deepsea_features[feature] > 1:
            deepsea_features[feature] -= 1
            feature = "{0}|{1}".format(feature, deepsea_features[feature])
        elif feature in deepsea_features and deepsea_features[feature] == 1:
            deepsea_features[feature] -= 1
        elif feature in deepsea_features and deepsea_features[feature] <= 0:
            continue

        all_features.append(feature)
        dfs_to_concat.append(_format_feature_dataset(
            filepath, feature))

    # handle edge case, this file isn't listed in the metadata file
    # that we have
    filepath = os.path.join(
        ENCODE_TF,
        "wgEncodeAwgTfbsSydhHepg2Srebp1InslnUniPk.narrowPeak.gz")
    feature = "HepG2|SREBP1|insulin"
    all_features.append(feature)
    dfs_to_concat.append(_format_feature_dataset(
        filepath, feature))
    deepsea_features[feature] -= 1

    ChIP_agg = pd.concat(dfs_to_concat, ignore_index=True)
    ChIP_agg.sort_values([0, 1, 2], ascending=True, inplace=True)
    print(ChIP_agg.head())

    dfs_to_concat = []

    # Roadmap Epigenomic features (DNase, histone marks)
    for index, filename in enumerate(os.listdir(Roadmap_Epi)):
        if ".narrowPeak.gz" not in filename:
            continue
        filepath = os.path.join(Roadmap_Epi, filename)
        filename = filename[:-len(".narrowPeak.gz")]
        EID, info = filename.split("-")

        row = Roadmap_metadata.loc[EID]
        # handling the edge cases
        cell_type = row.get("DONOR / SAMPLE ALIAS")
        if cell_type == "RO01746":
            cell_type = "Monocytes-CD14+RO01746 "
        if cell_type == "Osteobl":
            cell_type = "Osteoblasts"
        if "hESC-01" in cell_type:
            cell_type = "H1-hESC"

        if info == "H2A.Z":
            info = "H2AZ"

        feature = "{0}|{1}|None".format(cell_type, info)

        if feature not in deepsea_features:
            continue
        all_features_with_dups.append(feature)
        if feature in deepsea_features and deepsea_features[feature] > 1:
            deepsea_features[feature] -= 1
            feature = "{0}|{1}".format(feature, deepsea_features[feature] + 1)
        elif feature in deepsea_features and deepsea_features[feature] == 1:
            deepsea_features[feature] -= 1
        elif feature in deepsea_features and deepsea_features[feature] <= 0:
            continue

        all_features.append(feature)
        dfs_to_concat.append(_format_feature_dataset(
            filepath, feature))

    EID_agg = pd.concat(dfs_to_concat, ignore_index=True)
    EID_agg.sort_values([0, 1, 2], ascending=True, inplace=True)
    print(EID_agg.head())

    # concat all dataframes to make one unsorted BED file
    full_aggregate_file = pd.concat(
        [DNase_agg, ChIP_agg, EID_agg], ignore_index=True)
    full_aggregate_file.columns = ["chrom", "start", "end", "strand", "feature", "metadata_index"]
    output_file = os.path.join(output_dir, "deepsea_full_unsorted.bed")
    with open(output_file, 'w+') as file_handle:
        for row in full_aggregate_file.itertuples():
            file_handle.write("{0}\t{1}\t{2}\t{3}\n".format(row.chrom, row.start, row.end, row.feature))

    print("Total number of features: {0}".format(len(all_features)))
    output_features = os.path.join(output_dir, "distinct_features.txt")
    with open(output_features, 'w+') as file_handle:
        features = sorted(all_features)
        for f in features:
            file_handle.write("{0}\n".format(f))

