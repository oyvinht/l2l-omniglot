#!/usr/bin/env bash

JUBE=0
VENV=`pwd`/venv3
GENN=`pwd`/genn
PYNN=`pwd`/pynn_genn
L2L=`pwd`/L2L
NVCC=`which nvcc`
CUDA=`echo $NVCC | sed 's/\/bin\/nvcc//g'`


python3 -m venv $VENV

echo "export CUDA_PATH=$CUDA" >> $VENV/bin/activate

source $VENV/bin/activate

pip install --upgrade pip
pip install Cython
pip install numpy
pip install scipy
pip install matplotlib
pip install setuptools
pip install scikit-learn
pip install skutil
pip install pyyaml

git clone https://github.com/genn-team/genn $GENN
cd $GENN

make LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
make DYNAMIC=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
make DYNAMIC=True MPI_ENABLE=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
make MPI_ENABLE=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/


make CPU_ONLY=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
make CPU_ONLY=True DYNAMIC=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
make CPU_ONLY=True MPI_ENABLE=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
make CPU_ONLY=True DYNAMIC=True MPI_ENABLE=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/

sed -i 's/numpy>1\.6, < 1\.15/numpy>1\.6/g' setup.py
python setup.py develop

echo "export PATH=\$PATH:$GENN/bin" >> $VENV/bin/activate

git clone https://github.com/genn-team/pynn_genn $PYNN
cd $PYNN
sed -i 's/numpy>1\.6, < 1\.15/numpy>1\.6/g' setup.py
python setup.py develop


git clone https://github.com/chanokin/L2L $L2L
cd $L2L
if [ $JUBE -eq 1 ]
then
    ./install.sh $VENV/bin/activate
else
    python setup.py develop
fi

