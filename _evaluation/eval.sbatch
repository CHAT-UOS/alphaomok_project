#!/bin/bash
#SBATCH -p gpu2
#SBATCH -c 32
#SBATCH --gres=gpu:a10:2
#SBATCH -o ./_out/%j.%N.out
#SBATCH -e ./_err/%j.%N.err

python eval_main_4.py