from sequences.genome import Genome
from targets.genomic_features import GenomicFeatures
import h5py
import gzip

class UpdateSeqweaver():
    """
    Stores an updated dataset specifying sequence regions and features.
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
    def __init__(self, input_path, output_path, feature_path, sequence_len=1000):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.feature_path = feature_path
        self.sequence_len = sequence_len

        with open(self.feature_path, 'r') as handle:
            self.feature_set = [line.split('\n') for line in handle.readlines()]

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
                list_of_regions.append(line.strip().split())

        seqs = Genome(hg19_fasta, blacklist_regions = 'hg19')
        targets = GenomicFeatures(bed_file,
                  features = self.feature_set, feature_thresholds = 0.5)

        with h5py.open(self.output_path) as fh:
            training_seqs = []
            training_labels = []
            for r in list_of_regions:
                chrom, start, end, strand, target = r
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

            fh.write("sequences", data=np.vstack(training_seqs))
            fh.write("targets", data=np.vstack(training_labels))