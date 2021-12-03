
'''
Pip installs needed and working on google colab -
!pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
!pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
!pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
!pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
!pip install torch-geometric -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html

To use the pretrained models, might have to manually add a '.zip' while reading the model.
'''


import torch
# import torch_geometric
from torch_geometric.datasets import Planetoid
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
from torch_geometric.nn import GAT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

dataset = Planetoid(root="./",name= "Cora",num_train_per_class=50,split='random')
d = dataset.data
x = dataset.data.x
N = x.shape[0]
adj = torch.zeros((N,N),dtype=torch.int16)


for e in d.edge_index.T:
    adj[e[0]][e[1]] += 1
edges = [ [] for _ in range(N) ]

i = 0
for e in d.edge_index.T:
    i+=1
    edges[e[0].item()].append(e[1].item())

def gen_step_rw(idx, edges, steps):
    l = []
    for i in range(steps):
        sz = len(edges[idx])
        if(sz==0):
            nxt = idx
        else:
            nxt = random.randint(0,sz-1)
        l.append(edges[idx][nxt])
        idx = edges[idx][nxt]
    return l

def find_RWR(x, edges, idx, k = 1000, steps = 4):
    N = x.shape[0]
    rwr_idx = torch.zeros((N))
    for i in range(k):
        # print(i,idx)
        l = gen_step_rw(idx, edges, steps)
        for idx2 in l:
            rwr_idx[idx2]+=1
    rwr_idx /= (steps*k)
    return rwr_idx

RWR_distances = torch.zeros((N,N))
for i in range(N):
    RWR_distances[i] = find_RWR(x, edges, i)
    if(i%500==0) : print(i)


INP_DIM = x.shape[1]
in_features = INP_DIM
DIM = 128
tms = 350

def loss_calc(pred, actual):
    return torch.sqrt( torch.sum(torch.square(pred-actual)) / pred.shape[0] ).item()
# loss_calc(test_op,test_actual_op)

class GATLayer(nn.Module):
    def __init__(self, model, INP_DIM=INP_DIM, DIM=DIM, concat=True):
        super(GATLayer, self).__init__()
        self.dropout       = 0.6        # drop prob = 0.6
        self.in_features   = INP_DIM    # 
        self.out_features  = DIM
        self.hidden_layers = 100
        self.alpha         = 0.2          # LeakyReLU with negative input slope, alpha = 0.2
        self.model = model
        
        # Embedding learn
        self.W = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, input):
        # Linear Transformation
        h1 = torch.mm(input[:,0,:], self.W) # matrix multiplication
        h2 = torch.mm(input[:,1,:], self.W) # matrix multiplication
        N = h1.size()[0]
        
        h = torch.cat((h1, h2), dim=1).view(N, 2 * self.out_features)
        op = self.model(h,dataset.data.edge_index)
        op = op.squeeze(1)
        return op

def gen_train_data(idx):
    rd = random.sample(range(0, tms), 31)
    l = []
    for i in range(31):
        if(rd[i]!=idx):
            l.append(rd[i])
        if(len(l)==30): break
    return l
    
def gen_test_data(idx):
    rd = random.sample(range(0, 1000), 11)
    l = []
    for i in range(11):
        if(rd[i]!=idx):
            l.append(rd[i])
        if(len(l)==10): break
    return l

if sys.argv[1]=='0':
    train_data = torch.zeros((tms*30, 2, INP_DIM))
    train_op = torch.zeros((tms*30))

    r=torch.randperm(train_data.shape[0])
    train_data = train_data[r]
    train_op = train_op[r]

    train_data_all = dataset.data.x[dataset.data.train_mask]
    train_data_mask = [i for i in range(len(dataset.data.train_mask)) if dataset.data.train_mask[i]]

    for i in range(tms):
        l = gen_train_data(i)
        for j in range(30):
            train_data[i*30 + j][0][:] = train_data_all[i]
            train_data[i*30 + j][1][:] = train_data_all[l[j]]
            train_op[i*30 + j] = RWR_distances[train_data_mask[i],train_data_mask[l[j]]]

    DIM = 128
    gatmodel = GAT(in_channels = 2*DIM,
                hidden_channels = DIM,
                num_layers = 3,
                out_channels = 1,
                dropout = 0.0,
                )

    l = []

    model = GATLayer(gatmodel).to(device)
    data = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, train_op)
        
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), './model2')
    torch.save(gatmodel.state_dict(), './gatmodel2')

elif sys.argv[1]=='1':
    gatmodel = GAT(in_channels = 2*DIM,hidden_channels = DIM,num_layers = 3,out_channels = 1,dropout = 0.0,)
    gatmodel.load_state_dict(torch.load('./gatmodel'))
    model = GATLayer(gatmodel).to(device)
    model.load_state_dict(torch.load('./model'))

test_data_all = dataset.data.x[dataset.data.test_mask]
test_data_mask = [i for i in range(len(dataset.data.test_mask)) if dataset.data.test_mask[i]]
test_data = torch.zeros((1000*10, 2, INP_DIM))
test_actual_op = torch.zeros((1000*10))

for i in range(1000):
    l = gen_test_data(i)
    for j in range(10):
        test_data[i*10 + j][0][:] = test_data_all[i]
        test_data[i*10 + j][1][:] = test_data_all[l[j]]
        test_actual_op[i*10 + j] = RWR_distances[ test_data_mask[i] , test_data_mask[ l[j] ] ]

model.eval()
test_op = model(test_data)
ls = loss_calc(test_op,test_actual_op)
print(ls)
