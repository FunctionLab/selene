#!/bin/bash
set -o errexit

original_dir=$(pwd)

# MOUSE ENCODE DATA

mouse_data_dir="./data/mouse_data/"
cd $mouse_data_dir

mm10_ENCODE_fasta=$mouse_data_dir"/mm10_no_alt_analysis_set_ENCODE.fasta"
mm10_ENCODE_fasta_gz=$mm10_ENCODE_fasta".gz"
gunzip $mm10_ENCODE_fasta_gz
samtools faidx $mm10_ENCODE_fasta

genomic_features_dir=$mouse_data_dir"/ENCODE_mouse_data"
mkdir -p $genomic_features_dir
cd $genomic_features_dir
xargs -n 1 curl -0 -L < ../ENCODE_mouse_data_files.txt

cd ..
bigBed_files="./bigBed_filenames.txt"
bigBed_dir="./ENCODE_mouse_bigBed_data"
find $genomic_features_dir -name \*.bigBed > $bigBed_files
while read i
    do bigBedToBed "$i" $bigBed_dir/"$i".bed
done <$bigBed_files

aggregate_output_file="./mm10_aggregate_unsorted.bed"
python ./mm10_aggregate_data_file.py $genomic_features_dir \
                                     $bigBed_dir \
                                     $aggregate_output_file

$sorted_aggregate_file="mm10_sorted_aggregate.bed"
$sorted_aggregate_file_gz=$sorted_aggregate_file".gz"
sort -k1V -k2n -k3n $aggregate_output_file > $sorted_aggregate_file
bgzip -c $sorted_aggregate_file > $sorted_aggregate_file_gz
tabix -p bed $sorted_aggregate_file_gz

# HUMAN ENCODE DATA

cd $original_dir

human_data_dir="./data/human_data"
cd $human_data_dir

hg19_ENCODE_fasta=$human_data_dir"/male.hg19.fasta"
hg19_ENCODE_fasta_gz=$hg19_ENCODE_fasta".gz"
gunzip $hg19_ENCODE_fasta_gz
samtools faidx $hg19_ENCODE_fasta

# where did each of the human ENCODE datasets come from?
#genomic_features_dir=$human_data_dir"/ENCODE_human_data"
#mkdir -p $genomic_features_dir

