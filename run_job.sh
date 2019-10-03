#!/usr/bin/env bash
#SBATCH --account=hhd34
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --mail-user=g.pineda-garcia@sussex.ac.uk
#SBATCH --mail-type=ALL

source venv3/bin/activate

python3 fexplorer.py
