How to run the program (Q2) -

For Q2 - 
$ bash Q2.sh <saved>

Modules requried -
-> Pytorch, Pytorch-geometric
which need few other installations as mentioned below-

We are assuming following python packages been installed on the machine that would be running 
our code as HPC didn't had available modules for these.

We used the following commands to run the code on google colab-

a. pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
b. pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
c. pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
d. pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
e. pip install torch-geometric -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.htm

