"""
This is the script used to create the BED files with gencode protein-coding
genes. Posted here for reproducibility.

TODO: add documentation, switch to docopt.
"""
import os
import sys


if __name__ == "__main__":
    filename = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)
    bed_file = os.path.join(
        output_dir, "protein_coding_l12_genes.bed")
    with open(filename, 'r') as gtf_file, \
            open(bed_file, 'w+') as write_file:
        for line in gtf_file:
            if line.startswith('#'):
                continue
            fields = line.strip('\n').split('\t')
            if fields[2] != "gene":
                continue

            info = dict(x.strip().split() for x in fields[8].split(';') if x != '')
            info = {k: v.strip('"') for k, v in info.items()}

            if info["gene_type"] != "protein_coding":
                continue
            if info["level"] != "2" and info["level"] != "1":
                continue

            write_file.write(
                "{0}\t{1}\t{2}\t{3}\t{4}\n".format(
                    fields[0],
                    fields[3],
                    fields[4],
                    fields[5],
                    info["gene_name"]))
