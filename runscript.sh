#!/bin/bash
#SBATCH -N 1
#SBATCH --time=02:00:00
#SBATCH --mem=50000
#SBATCH --gres=gpu:1
#SBATCH -o o_seqmodel_tgpu_%j.out
#SBATCH -e e_seqmodel_tgpu_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kchen@flatironinstitute.org

module load cudnn/cuda-8.0/6.0
module load anaconda3/4.4.0

export myscratch="/scratch/kc31/job"
data_dir="/tigress/kc31/data_small"

rm -rf $myscratch
mkdir -p $myscratch

source activate mazu

python -u seq_model.py $data_dir"/mm10_no_alt_analysis_set_ENCODE.fasta" \
                    $data_dir"/reduced_agg_beds_1.bed" \
                    $data_dir"/reduced_agg_beds_1.bed.gz" \
                    $myscratch"/output9317.model" \
                    --holdout-chrs=chr8,chr9 --radius=100 --window=1001 \
                    --random-seed=123 --mode=train --use-cuda

cp -r $myscratch /tigress/kc31/
rm -rf $myscratch
