from sequences.genome import Genome
from targets.genomic_features import GenomicFeatures
import h5py
import gzip

SEQ_LENGTH = 1000

def from_midpoint(start, end):
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
    seq_start = midpoint - np.floor(SEQ_LENGTH / 2.)
    seq_end = midpoint + np.ceil(SEQ_LENGTH / 2.)

    return seq_start, seq_end


def construct_training_data(output_file, colname_file):
    """
    Construct training dataset from bed file and write to output_file.

    Parameters
    ----------
    output_file : str
        Path to the output file for the constructed training data.
    colname_file : str
        Path to a .txt file containing newline-delimited feature names.
    """
    with open(colname_file, 'r') as handle:
        feature_set = [line.split('\n') for line in handle.readlines()]

    list_of_regions = []
    with gzip.open(bed_file) as f:
        for line in f:
            list_of_regions.append(line.strip().split())

    seqs = Genome(hg19_fasta, blacklist_regions = 'hg19')
    targets = GenomicFeatures(bed_file,
              features = feature_set, feature_thresholds = 0.5)

    with h5py.open(output_file) as fh:
        training_seqs = []
        training_labels = []
        for r in list_of_regions:
            chrom, start, end, tgt, strand = r
            sstart, ssend = from_midpoint(start, end)

            # 1 x 4 x 1000 bp
            # get_encoding_from_coords : converts sequence to one-hot-encoding for each of the 4 bases
            dna_seq, has_unk = seqs.get_encoding_from_coords_check_unk(chrom, sstart, ssend, strand=strand)
            if has_unk: continue

            # 1 x n_features
            # get_feature_data: Computes which features overlap with the given region.
            labels = targets.get_feature_data(chrom, start, end, strand)
            training_seqs.append(dna_seq)
            training_labels.append(labels)

        # TODO: partition some to validation before writing
        fh.write("sequences", data=np.vstack(training_seqs))
        fh.write("targets", data=np.vstack(training_labels))