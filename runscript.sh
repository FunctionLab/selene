#!/bin/bash
#SBATCH -N 1
#SBATCH --time=120:00:00
#SBATCH --mem=75000
#SBATCH --gres=gpu:1
#SBATCH -o o_hgLR_%j.out
#SBATCH -e e_hgLR_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kchen@flatironinstitute.org

module load cudnn/cuda-8.0/6.0
module load anaconda3/4.4.0

hg19="/tigress/kc31/hg19/*"
data_dir_original="/tigress/kc31/compress_hg19_latest/*"
export data_dir="/scratch/data_hg"

source activate mazu

rm -rf $data_dir
mkdir -p $data_dir
cp -r $data_dir_original $data_dir
cp -r $hg19 $data_dir

python -u train_model.py deepsea SGD 0.14 \
                         configs/paths.yml \
                         configs/train_model.yml \
                         --runs=1 --verbose
rm -rf $data_dir
