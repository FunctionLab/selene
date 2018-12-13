# File is already sorted
cut -f 1-3 33545_peaks.bed > GATA1_proery_bm.bed

sed -i "s/$/\tProery_BM|GATA1/" GATA1_proery_bm.bed

bgzip -c GATA1_proery_bm.bed > GATA1_proery_bm.bed.gz

tabix -p bed GATA1_proery_bm.bed.gz

python get_test_regions.py GATA1_proery_bm.bed \
                           GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
                           regulatory_mutations.fa \
                           --seq-len=1000 \
                           --n-samples=20 \
                           --holdouts=chr8,chr9 \
                           --seed=42
