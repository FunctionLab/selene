"""
This module provides the `UpdateSeqweaver` class, which wraps the master bed file
containing all of the features' binding sites parsed from CLIP-seq.
It supports new dataset construction and training for Seqweaver.

"""
import h5py
import gzip
import numpy as np
import sys

from sequences.genome import Genome
from targets.genomic_features import GenomicFeatures
from samplers.dataloader import H5DataLoader
from train_model import TrainModel
from utils.config import load_path
from utils.config_utils import parse_configs_and_run

class UpdateSeqweaver():
    """
    Stores a dataset specifying sequence regions and features.
    Accepts a tabix-indexed `*.bed` file with the following columns,
    in order:
        [chrom, start, end, strand, feature]

    Parameters
    ----------
    input_path : str
        Path to the tabix-indexed dataset. Note that for the file to
        be tabix-indexed, it must have been compressed with `bgzip`.
        Thus, `input_path` should be a `*.gz` file with a
        corresponding `*.tbi` file in the same directory.
    output_path : str
        Path to the output constructed-training data file.
    feature_path : str
        Path to a '\n'-delimited .txt file containing feature names.
    hg_fasta : str
        Path to an indexed FASTA file -- a `*.fasta` file with
        a corresponding `*.fai` file in the same directory. This file
        should contain the target organism's genome sequence.

    """
    def __init__(self, input_path, train_path, validate_path, feature_path, hg_fasta, yaml_path, VAL_PROP=0.01, sequence_len=1000):
        """
        Constructs a new `UpdateSeqweaver` object.
        """
        self.input_path = input_path
        self.train_path = train_path
        self.validate_path = validate_path
        self.feature_path = feature_path
        self.yaml_path = yaml_path
        self.VAL_PROP = VAL_PROP

        self.hg_fasta = hg_fasta

        self.sequence_len = sequence_len

        with open(self.feature_path, 'r') as handle:
            self.feature_set = [line.split('\n')[0] for line in handle.readlines()]

    def _from_midpoint(self, start, end):
        """
        Computes start and end of the sequence about the peak midpoint.

        Parameters
        ----------
        start : int
            The 0-based first position in the region.
        end : int
            One past the 0-based last position in the region.

        Returns
        -------
        seq_start : int
            Sequence start position about the peak midpoint.
        seq_end : int
            Sequence end position about the peak midpoint.
        """
        region_len = end - start
        midpoint = start + region_len // 2
        seq_start = midpoint - np.floor(self.sequence_len / 2.)
        seq_end = midpoint + np.ceil(self.sequence_len / 2.)

        return int(seq_start), int(seq_end)

    def construct_training_data(self):
        """
        Construct training dataset from bed file and write to output_file.

        Parameters
        ----------
        output_path : str
            Path to the output file for the constructed training data.
        colname_file : str
            Path to a .txt file containing newline-delimited feature names.

        """
        list_of_regions = []
        with gzip.open(self.input_path) as f:
            for line in f:
                line = [str(data,'utf-8') for data in line.strip().split()]
                list_of_regions.append(line)

        seqs = Genome(self.hg_fasta, blacklist_regions = 'hg19')
        targets = GenomicFeatures(self.input_path,
                  features = self.feature_set, feature_thresholds = 0.5)

        data_seqs = []
        data_labels = []
        for r in list_of_regions:
            chrom, start, end, target, strand = r
            start, end = int(start), int(end)
            sstart, ssend = self._from_midpoint(start, end)

            # 1 x 4 x 1000 bp
            # get_encoding_from_coords : Converts sequence to one-hot-encoding for each of the 4 bases
            dna_seq, has_unk = seqs.get_encoding_from_coords_check_unk(chrom, sstart, ssend, strand=strand)
            if has_unk:
                continue
            if len(dna_seq) != self.sequence_len:
                continue

            # 1 x n_features
            # get_feature_data: Computes which features overlap with the given region.
            labels = targets.get_feature_data(chrom, start, end, strand=strand)

            data_seqs.append(dna_seq)
            data_labels.append(labels)

        # partition some to validation before writing
        val_count = int(np.floor(self.VAL_PROP * len(data_seqs)))
        validate_seqs = data_seqs[:val_count]
        validate_labels = data_labels[:val_count]
        training_seqs = data_seqs[val_count:]
        training_labels = data_labels[val_count:]

        with h5py.File(self.validate_path, "w") as fh:
            fh.create_dataset("valid_sequences", data=np.array(validate_seqs, dtype=np.int64))
            fh.create_dataset("valid_targets", data=np.array(validate_labels, dtype=np.int64))

        with h5py.File(self.train_path, "w") as fh:
            fh.create_dataset("train_sequences", data=np.array(training_seqs, dtype=np.int64))
            fh.create_dataset("train_targets", data=np.array(training_labels, dtype=np.int64))

    def _load_yaml(self):
        # load yaml configuration
        return load_path(self.yaml_path)

    def train_model(self):
        # load config file and train model
        yaml_config = self._load_yaml()
        parse_configs_and_run(yaml_config)