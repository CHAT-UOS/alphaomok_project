#!/bin/bash
#SBATCH -p gpu2
#SBATCH -c 32
#SBATCH --gres=gpu:a10:2
#SBATCH -o ./_base_out/%j.%N.out
#SBATCH -e ./_base_err/%j.%N.err

python main.py