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

rm -rf $myscratch
mkdir -p $myscratch

source activate mazu

python seqmodel.py

cp -r $myscratch /tigress/kc31/
rm -rf $myscratch
