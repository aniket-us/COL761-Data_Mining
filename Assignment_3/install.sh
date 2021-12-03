#!/bin/sh

# git clone https://github.com/krdipen/COL761-Assignments.git

# module load compiler/gcc/9.1.0
# module load compiler/python/3.6.0/ucs4/gnu/447
# module load pythonpackages/3.6.0/matplotlib/3.0.2/gnu
# module load pythonpackages/3.6.0/numpy/1.16.1/gnu
# module load pythonpackages/3.6.0/scikit-learn/0.21.2/gnu
# module load pythonpackages/3.6.0/pandas/0.23.4/gnu
# module load pythonpackages/3.6.0/scipy/1.1.0/gnu 
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html