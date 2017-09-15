#!/bin/bash
set -o errexit

data_dir="./data/mouse_data/"
cd $data_dir

mm10_ENCODE_fasta=$data_dir"/mm10_no_alt_analysis_set_ENCODE.fasta"

mm10_ENCODE_fasta_gz=$mm10_ENCODE_fasta".gz"
gunzip $mm10_ENCODE_fasta_gz

samtools faidx $mm10_ENCODE_fasta

genomic_features_dir=$data_dir"/ENCODE_mouse_data"
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

aggregate_output_file=$data_dir"/mm10_aggregate_unsorted.bed"
python ./mm10_aggregate_data_file.py $genomic_features_dir \
                                     $bigBed_dir \
                                     $aggregate_output_file


