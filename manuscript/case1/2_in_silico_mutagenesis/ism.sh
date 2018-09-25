#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -n 1
#SBATCH --mem 500000
#SBATCH -o ism_%j.out
#SBATCH -e ism_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kchen@flatironinstitute.org

source activate selene-env

python -u ../../../selene_cli.py ./in_silico_mutagenesis.yml
