#!/bin/bash

# Run this after running ../download_data.sh

python process_human_ENCODE.py chromatin_profiles/deepsea__919_features_.txt \
                               chromatin_profiles/ENCODE_DNase/ \
                               chromatin_profiles/ENCODE_TF/ \
                               chromatin_profiles/Roadmap_Epigenomics/ \
                               .

sort -k1V -k2n -k3n deepsea_full_unsorted.bed > sorted_deepsea_data.bed

bgzip -c sorted_deepsea_data.bed > sorted_deepsea_data.bed.gz

tabix -p bed sorted_deepsea_data.bed.gz

sort -o distinct_features.txt distinct_features.txt
