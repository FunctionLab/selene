#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -o variants_%j.out
#SBATCH -e variants_%j.err

source activate selene-env

python -u ../../../selene_cli.py ./variant_effect_prediction.yml
