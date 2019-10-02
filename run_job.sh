#!/usr/bin/env bash
#module load tmux Python/3.6.8 CUDA NCCL cuDNN unzip git HDF5 JUBE SciPy-Stack h5py scikit

BUDGET=hhd34
PROJ=chhd34
jutil env activate -p $PROJ -A $BUDGET

#SBATCH --account=$BUDGET
#SBATCH --partition=gpus
#SBATCH --nodes=5
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --mail-user=g.pineda-garcia@sussex.ac.uk
#SBATCH --mail-type=ALL


source venv3/bin/activate

python fexplorer.py
