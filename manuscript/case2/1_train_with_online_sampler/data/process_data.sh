#!/bin/bash

# Chromatin profiles download. ENCODE/Roadmap Epigenomics .bed files used
# in DeepSEA (Zhou & Troyanskaya, 2015).
wget https://zenodo.org/record/2214970/files/chromatin_profiles.tar.gz
tar -xzvf chromatin_profiles.tar.gz

python process_chromatin_profiles.py chromatin_profiles/deepsea__919_features_.txt \
                                     chromatin_profiles/ENCODE_DNase/ \
                                     chromatin_profiles/ENCODE_TF/ \
                                     chromatin_profiles/Roadmap_Epigenomics/ \
                                     .

sort -k1V -k2n -k3n deepsea_full_unsorted.bed > sorted_deepsea_data.bed

bgzip -c sorted_deepsea_data.bed > sorted_deepsea_data.bed.gz

tabix -p bed sorted_deepsea_data.bed.gz

sort -o distinct_features.txt distinct_features.txt

python create_TF_intervals.py distinct_features.txt \
                              sorted_deepsea_data.bed \
                              TF_intervals_unmerged.txt

bedtools merge -i TF_intervals_unmerged.txt > TF_intervals.txt
