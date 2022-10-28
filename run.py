import torch as th
import torch.nn as nn
import geoopt as gt
import torch
import torch.nn.functional as F

from scipy.io import arff
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings; warnings.simplefilter('ignore')
from torch.utils.data import DataLoader
from model import *

import sys
sys.path.append('..') 

default_dtype = th.float64
th.set_default_dtype(default_dtype)

if th.cuda.is_available():
    cuda_device = th.device('cuda:1')
    th.cuda.set_device(device=cuda_device)
else:
    raise Exception('No CUDA device found.')

import argparse

parser = argparse.ArgumentParser(
    description="Embedding"
)
parser.add_argument(
    "--dataset", default="derisi_FUN", choices=["expr_FUN", "derisi_FUN", "cellcycle_FUN", "spo_FUN", "expr_GO", "derisi_GO", "cellcycle_GO", "spo_GO", "Diatoms","Enron_corr","ImCLEF07A", "ImCLEF07D"], help="dataset"
)
parser.add_argument(
    "--learning_rate", default=1e-3, type=float, help="Learning rate"
)

args = parser.parse_args()
dataset = args.dataset

sampling_rate = 0.3

path = './data/'
num_labels = {"expr_FUN":500, "derisi_FUN":500, "cellcycle_FUN":500, "spo_FUN":500, "expr_GO":4132, "derisi_GO":4132, "cellcycle_GO":4132, "spo_GO":4132, "Diatoms":399,"Enron_corr":57,"ImCLEF07A":97, "ImCLEF07D":47}
num_label = num_labels[args.dataset]

def load_data(path, dataset, d='train', num_label=num_label,):
    data = arff.loadarff(path + dataset +'/'+d+'-normalized.arff')
    df = pd.DataFrame(data[0])
    return df.values[:,0:-num_label].astype(float), df.values[:,-num_label:].astype(int), list(df.columns)[-num_label:]


X_train, Y_train, label_list = load_data(path, dataset, d='train',num_label=num_label)
X_dev, Y_dev,_ = load_data(path, dataset, d='dev',num_label=num_label)
X_test, Y_test,_ = load_data(path, dataset, d='test',num_label=num_label)

feature_dim = X_train.shape[1]
label_dim = Y_train.shape[1]
print("number of feature and labels:", feature_dim, label_dim)

# full_graph = pd.read_csv(path + dataset +'/' + 'hierarchy_tc.edgelist', sep=' ', header=None).values[:,0:2]
# tc_graph = pd.read_csv(path + dataset +'/' + 'hierarchy.edgelist', sep=' ', header=None).values[:,0:2]

# indexing = lambda t: label_list.index(t)
# vfunc = np.vectorize(indexing)
# full_graph = vfunc(full_graph)
# tc_graph = vfunc(tc_graph)
# full_graph.shape,tc_graph.shape

# exclusion_from_instances = []
# for i in range(num_label):
#     for j in range(i,num_label):
#         if np.count_nonzero( Y_train[:,i] & Y_train[:,j]) == 0:
#             exclusion_from_instances.append([i,j])


# sublabels = {}
# direct_parent = {}

# for pair in full_graph:
#     child, parent = pair[0],pair[1]
#     if sublabels.get(parent) == None:
#         sublabels[parent] = []
#     if sublabels.get(child) == None:
#         sublabels[child] = []
#     sublabels[parent].append(child)

# direct_parent[label_list.index('root')] = -1
# for pair in tc_graph:
#     child, parent = pair[0],pair[1]
    
#     direct_parent[child] = parent

# implication = tc_graph
# exclusion = []

# for a in range(len(label_list)):
#     for b in range(a, len(label_list)):
#         overlap = list(set(sublabels[a]) & set(sublabels[b]))
#         if len(overlap) == 0 and direct_parent[a] == direct_parent[b]:
#             if np.count_nonzero(Y_train[:,a] & Y_train[:,b]) == 0: 
#                 exclusion.append([a,b])

# full_implication = torch.tensor(implication).to(cuda_device)    
# full_exclusion = torch.tensor(exclusion).to(cuda_device)   
# print(full_implication.shape[0],full_exclusion.shape[0])
# implication=full_implication[torch.randperm(full_implication.size()[0])][0:int(full_implication.size()[0]*sampling_rate)]
# exclusion=full_exclusion[torch.randperm(full_exclusion.size()[0])][0:int(full_exclusion.size()[0]*sampling_rate)]
# print(implication.shape[0],exclusion.shape[0])

implication = torch.load(path + dataset +'/implication.pt')
exclusion = torch.load(path + dataset +'/exclusion.pt')


from util.metric import *
from util.pytorchtools import *

lr = 1e-3
hidden_size = 32
embed_dim = 32
patience = 20
batch_size = 4

model = HMI(feature_dim, hidden_size, embed_dim, label_dim)
model.to(cuda_device)
loss = nn.BCEWithLogitsLoss()
optim = gt.optim.RiemannianAdam(model.parameters(), lr=lr)
early_stopping = EarlyStopping(patience=patience, verbose=False)

X_train_dataloader, Y_train_dataloader = DataLoader(torch.tensor(X_train).to(cuda_device), batch_size=batch_size, shuffle=True), DataLoader( torch.tensor(Y_train).to(cuda_device), batch_size=batch_size, shuffle=True)

X_dev, Y_dev = torch.tensor(X_dev).to(cuda_device), torch.tensor(Y_dev).to(cuda_device)
X_test, Y_test = torch.tensor(X_test).to(cuda_device), torch.tensor(Y_test).to(cuda_device)

def train(epoch, model):
    model.train()
    best_map = []
    best_cv = []
    for e in range(1, epoch + 1):
        # training block
        for X_train, Y_train in zip(X_train_dataloader,Y_train_dataloader):
            optim.zero_grad()
            logits, inside_loss, disjoint_loss, label_reg, instance_reg = model(X_train,implication,exclusion)
            classification_loss = loss(logits,Y_train.double())
            label_graph_loss = 1e-4*(inside_loss + disjoint_loss )
            train_loss = classification_loss + label_graph_loss
            train_loss.backward()
            optim.step()
        # validation block
        model.eval()    
        logits, _, _, _, _instance_reg = model(X_dev,implication,exclusion)
        val_loss = loss(logits,Y_dev.double())
        # testing block
        logits,_,_, _, _ = model(X_test,implication,exclusion)
        pred = F.sigmoid(logits) 
        mAP = mean_average_precision(pred, Y_test.double()) 
        best_map.append(mAP)
        
        if e%5 ==0:
            print(classification_loss.item(), inside_loss.item(),disjoint_loss.item())
            print('Epoch:%d\tTrainLoss:%.3f\tValLoss:%.3f\tmAP:%.3f' %(e, train_loss.item(),val_loss.item(),mAP))
        model.train()
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
#             print("Early stopping")
            break
    print(embed_dim,max(best_map))     


optim = gt.optim.RiemannianAdam(model.parameters(), lr=lr)


optim = gt.optim.RiemannianAdam(model.parameters(), lr=lr)

lr = 1e-3
hidden_size = 32
embed_dim = 64
patience = 20
batch_size = 4
epoch = 200
model = HMI(feature_dim, hidden_size, embed_dim, label_dim)
model.to(cuda_device)
loss = nn.BCEWithLogitsLoss()
optim = gt.optim.RiemannianAdam(model.parameters(), lr=lr)
early_stopping = EarlyStopping(patience=patience, verbose=False)

train(epoch, model)

# -----Run hyperparameter optimization--------
# for dim in [2,4,8,16,32,64,128,256,512]:
#     lr = 1e-3
#     hidden_size = 32
#     embed_dim = 64
#     patience = 20
#     batch_size = 4
#     epoch = 200
#     model = HMI(feature_dim, hidden_size, embed_dim, label_dim)
#     model.to(cuda_device)
#     loss = nn.BCEWithLogitsLoss()
#     optim = gt.optim.RiemannianAdam(model.parameters(), lr=lr)
#     early_stopping = EarlyStopping(patience=patience, verbose=False)
#     train(epoch, model)
        
    

        
        


   

