import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import sys
import time
import mtree_library
import math

def modify(arr,val,id):
    if(len(arr)<5): 
        i=len(arr)-1
        l=len(arr)
        arr.append((val,id))
        if(val>=arr[i][0]): 
            return
        while(val<arr[i][0] and i>=0):
            i-=1
        while(i<l):
            temp=arr[i+1]
            arr[i+1] = arr[l]
            arr[l] = temp  
            i+=1
        return 
    if(val >= arr[4][0]):
        return 
    i=4
    while(val<arr[i][0] and i>=0):
        i-=1
    temp=arr[i+1]
    arr[i+1] = (val,id)
    i+=1
    while(i<4):
        temp2 = arr[i+1]
        arr[i+1] = temp
        temp = temp2
        i+=1

data = pd.read_csv(sys.argv[1],header=None,sep=" ")
data = np.array(data)
data = data[:,:-1]

kd_mean = []
m_mean = []
seq_mean = []

kd_std = []
m_std = []
seq_std = []

def L2(x, y):
    ans = 0
    for i in range(len(x)):
        ans += (x[i]-y[i])**2
    return math.sqrt(ans)

def sequential(pc,z):
    m,n = pc.shape
    i=0
    arr = []
    for x in pc:
        dist = np.linalg.norm(x-z)
        modify(arr,dist,i)
        i+=1
    ans = []
    for x,y in arr:
        ans.append(y)
    return ans

dimensions = [2,4,10,20]
for d in dimensions:
    pca = PCA(n_components=d)
    pc = pca.fit_transform(data)
    m,n = pc.shape

    tree = mtree_library.MTree(L2, max_node_size=m)
    tree.add_all(pc.tolist())
    kd = KDTree(pc, metric='euclidean')

    # y=np.random.randint(100, size=(100,d))
    id = np.random.choice(m, 100)
    y = [pc[i] for i in id]


    k=5
    kd_time = []
    m_time = []
    seq_time = []
    for x in y:
        z=np.array([x])

        start_time = time.time()
        distances, indices = kd.query(z, k = k)
        kd_time.append(time.time()-start_time)

        start_time = time.time()        
        point = sequential(pc,z)
        seq_time.append(time.time()-start_time)

        start_time = time.time()        
        tree.search(x.tolist(),5)
        m_time.append(time.time()-start_time)

    kd_mean.append(np.mean(kd_time))
    kd_std.append(np.std(kd_time))
    seq_mean.append(np.mean(seq_time))
    seq_std.append(np.std(seq_time))
    m_mean.append(np.mean(m_time))
    m_std.append(np.std(m_time))


# print(kd_mean)
# print(kd_std)
# print(seq_mean)
# print(seq_std)
# print(m_mean)
# print(m_std)

plt.errorbar(dimensions, kd_mean, kd_std, label='kdtree')
plt.errorbar(dimensions, m_mean, m_std, label='mtree')
plt.errorbar(dimensions, seq_mean, seq_std, label='Sequencial search')

plt.title('Performance Comparision')
plt.ylabel('Average Query time (sec)', fontweight ='bold', fontsize = 15)
plt.xlabel('dimensions', fontweight ='bold', fontsize = 15)

plt.legend()
plt.savefig(sys.argv[2]+".png")
