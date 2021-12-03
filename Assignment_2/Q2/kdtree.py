import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import sys


def reduce_dimension(data,d):
    pca = PCA(n_components=d)
    pc = pca.fit_transform(data)
    return pc


def crete_tree(pc):
    kd = KDTree(pc, metric='euclidean')
    return kd

def query(kd,z,k):
    distances, indices = kd.query(z, k = k)
    return distances,indices


if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1],header=None,sep=" ")
    data = np.array(data)
    data = data[:,:-1]

    d=2
    if(len(sys.argv)>2): d=int(sys.argv[2])
    pc = reduce_dimension(data,d)
    m,n = pc.shape

    kd = crete_tree(pc)

    k=5
    if(len(sys.argv)>3): k=int(sys.argv[3])
    id = np.random.choice(m,2)
    y = [pc[i] for i in id]

    for x in y:
        z=np.array([x])
        dist, ind = query(kd,z,k)
        ans = [pc[i] for i in ind]
        print(k,"-NN of point",x,"are:")
        print(ans)
