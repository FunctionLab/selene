#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -o ism_%j.out
#SBATCH -e ism_%j.err

source activate selene-env

python -u ../../../selene_cli.py ./in_silico_mutagenesis.yml
