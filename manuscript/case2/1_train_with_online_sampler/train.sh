#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --constraint=v100
#SBATCH -o train_online_%j.out
#SBATCH -e train_online_%j.err

source activate selene-env

python -u ../../../selene_cli.py ./train_online_sampler.yml --lr=0.08
