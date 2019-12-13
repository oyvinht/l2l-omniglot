#!/usr/bin/env bash

#SBATCH --account=hhd34
#SBATCH --partition=gpus
#SBATCH --nodes=25
#SBATCH --ntasks=100
### #SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --threads-per-core=1

#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mail-user=g.pineda-garcia@sussex.ac.uk
#SBATCH --mail-type=ALL

####  # --exclusive

source venv3/bin/activate

python3 fexplorer.py
