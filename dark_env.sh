#!/bin/bash
module purge
module load anaconda3/2020.2
conda create --name dark-env --channel conda-forge --channel pytorch python=3.7 pytorch=1.4 torchvision \
cudatoolkit=10.1 pytables matplotlib cython h5py pyfftw notebook -y
conda activate dark-env

module load rh/devtoolset/8 cudatoolkit/10.1
CUDA=cu102
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric

pip install line-profiler
pip install ipdb

module load openmpi/gcc/3.1.5/64
export MPICC=$(which mpicc)
pip install mpi4py

git clone https://github.com/franciscovillaescusa/Pylians3.git
cd Pylians3/library
python setup.py install
