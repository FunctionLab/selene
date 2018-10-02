#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --constraint=v100
#SBATCH -n 1
#SBATCH -o train_mats_%j.out
#SBATCH -e train_mats_%j.err

source activate selene-env

python -u ../../../selene_cli.py ./train_deepsea_mat.yml --lr=0.08
