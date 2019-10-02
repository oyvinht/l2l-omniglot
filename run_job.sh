#!/usr/bin/env bash
#module load tmux Python/3.6.8 CUDA NCCL cuDNN unzip git HDF5 JUBE SciPy-Stack h5py scikit

#BUDGET=JUWELS_GPUS
PROJ=chhd34
jutil env activate -p $PROJ #-A $PROJ

#SBATCH --account=$PROJ
#SBATCH --partition=gpus
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=9
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1

source venv3/bin/activate

python fexplorer.py
