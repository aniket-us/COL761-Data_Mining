import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import sys
import mtree_library
import math

def L2(x, y):
    ans = 0
    for i in range(len(x)):
        ans += (x[i]-y[i])**2
    return math.sqrt(ans)

def reduce_dimension(data,d):
    pca = PCA(n_components=d)
    pc = pca.fit_transform(data)
    return pc


def crete_tree(pc,m):
    tree = mtree_library.MTree(L2, max_node_size=m)
    tree.add_all(pc.tolist())
    return tree

def query(tree,x,k):
    tree.search(x.tolist(),k)    

if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1],header=None,sep=" ")
    data = np.array(data)
    data = data[:,:-1]

    d=2
    if(len(sys.argv)>2): d=int(sys.argv[2])
    pc = reduce_dimension(data,d)
    m,n = pc.shape

    tree = crete_tree(pc,m)

    k=5
    if(len(sys.argv)>3): k=int(sys.argv[3])
    id = np.random.choice(m,2)
    y = [pc[i] for i in id]

    for x in y:
        query(tree,x,k)


