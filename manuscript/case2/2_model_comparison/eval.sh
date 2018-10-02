#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH -o eval_%j.out
#SBATCH -e eval_%j.err

source activate selene-env

python -u ../../../selene_cli.py ./evaluate.yml
