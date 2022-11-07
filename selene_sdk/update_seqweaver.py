"""
This module provides the `UpdateSeqweaver` class, which wraps the master bed file
containing all of the features' binding sites parsed from CLIP-seq.
It supports new dataset construction and training for Seqweaver.

"""
from sequences.genome import Genome
from targets.genomic_features import GenomicFeatures
import h5py
import gzip
import numpy as np

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

    """
    def __init__(self, input_path, output_path, feature_path, hg_fasta, sequence_len=1000):
        """
        Constructs a new `UpdateSeqweaver` object.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.feature_path = feature_path
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

        return seq_start, seq_end

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

        with h5py.File(self.output_path, "w") as fh:
            training_seqs = []
            training_labels = []
            for r in list_of_regions:
                chrom, start, end, strand, target = r
                start, end = int(start), int(end)
                sstart, ssend = self._from_midpoint(start, end)

                # 1 x 4 x 1000 bp
                # get_encoding_from_coords : Converts sequence to one-hot-encoding for each of the 4 bases
                dna_seq, has_unk = seqs.get_encoding_from_coords_check_unk(chrom, sstart, ssend, strand=strand)
                if has_unk: continue

                # 1 x n_features
                # get_feature_data: Computes which features overlap with the given region.
                labels = targets.get_feature_data(chrom, start, end, strand)
                training_seqs.append(dna_seq)
                training_labels.append(labels)

            fh.create_dataset("sequences", data=np.vstack(training_seqs))
            fh.create_dataset("targets", data=np.vstack(training_labels))