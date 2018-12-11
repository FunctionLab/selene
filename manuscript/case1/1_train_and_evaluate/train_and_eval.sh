#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH -n 1
#SBATCH --mem=500000
#SBATCH -o train_%j.out
#SBATCH -e train_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<EMAIL>

source activate selene-env

python -u ../../../selene_cli.py ./train_and_eval.yml --lr=0.01
