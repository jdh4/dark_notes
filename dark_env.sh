#!/bin/bash
module load anaconda3/2020.2
conda create --name dark-env pytorch torchvision cudatoolkit=10.2 matplotlib --channel pytorch -y
conda activate dark-env
module load rh/devtoolset/8 cudatoolkit/10.2 
CUDA=cu102
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric
pip install line-profiler
