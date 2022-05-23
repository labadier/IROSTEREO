# %% make graphs and get betweenness
import proto
import torch, numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx
import matplotlib.pyplot as plt
from utils.utils import bcolors
from torch_geometric.data import Data as GData
from torch_geometric.loader import DataLoader as GLoader


def PrepareGraph(data, beta=0.97, avecPrototypes = True, protoKind = 'centrality') -> tuple:
  
  mark = None
  def dfs( node, level ):
  
    mark[node] = level
    if level == 3:
      return
      
    for ady in G[node]:
      if mark[ady] != -1 and mark[ady] <= mark[node]:
        continue
      dfs(ady, level + 1)

  with torch.no_grad():

    alpha = [cosine_similarity(i) for i in data['sem_encodings']]
    edge_index = [list(zip(*np.where(graphs>beta))) for graphs in alpha]  #**

    mask = torch.zeros(data['sem_encodings'].shape[:-1])
    if avecPrototypes:
      print(f"{bcolors.OKGREEN}{bcolors.BOLD}Computing Prototypes{bcolors.ENDC}:", end="") 
      
      perc = 0
      for index, graph in enumerate(edge_index):

        if (index+1)/len(edge_index) - perc >= 0.0001:
          perc = (index+1)/len(edge_index)
          print(f"\r{bcolors.OKGREEN}{bcolors.BOLD}Computing Prototypes{bcolors.ENDC}: {perc*100.0:.2f}%", end="") 

        G = networkx.Graph(graph)
        comp = [sorted(c) for c in next(networkx.community.girvan_newman(G))]
        comp_index = {k:v for v in range(len(comp)) for k in comp[v] }

        G = networkx.Graph([i for i in G.edges if comp_index[i[0]] == comp_index[i[1]]])
        bw = networkx.betweenness_centrality(G)

        if protoKind == 'centrality':

          Edegree = [np.mean([G.degree[j] for j in comp[i]]) for i in range(len(comp))]
          Prototypes = [int(np.ceil(np.log(len(comp[i]))/np.log(Edegree[i]))) + (Edegree[i] <= 2) for i in range(len(comp))]
          comp = [sorted([(node, bw[node]) for node in comp[i]], reverse=True, key = lambda x : x[1])[:Prototypes[i]] for i in range(len(comp))]
          mask[index][[node[0] for c in comp for node in c]] = 1.0
        else:

          nodes = sorted([i for i in range(len(G.nodes))], reverse=True, key = lambda node : bw[node])
          for i in nodes:
            if mark[i] == -1 or mark[i] == 3:
              mask[index][i] = 1.0
              dfs(i, 1)

        mask[index] /= torch.sum(mask[index])
      
      mask = mask.unsqueeze(1)
      print(f"\r{bcolors.OKGREEN}{bcolors.BOLD}Computing Prototypes{bcolors.ENDC}: {perc*100.0:.2f}%") 
    
      edge_index = [torch.tensor(graph).t().contiguous() for graph in edge_index]
      return edge_index, mask, torch.tensor(data['labels'])

    edge_index = [torch.tensor(graph).t().contiguous() for graph in edge_index]
    return edge_index, torch.tensor(data['labels'])

data = torch.load('logs/raw/train_penc_en.pt')
data = {'encodings': data, 'labels':np.zeros((len(data),))}
e, m, x, y = PrepareGraph(data, beta=0.98)
#%%
beta = 0.97
alpha = [cosine_similarity(i) for i in data['encodings']]
edge_index = [list(zip(*np.where(graphs>beta))) for graphs in alpha]  #**

mask = torch.zeros(data['encodings'].shape[:-1])

print(f"{bcolors.OKGREEN}{bcolors.BOLD}Computing Prototypes{bcolors.ENDC}:", end="") 

perc = 0
for index, graph in enumerate(edge_index):

  if (index+1)/len(edge_index) - perc >= 0.0001:
    perc = (index+1)/len(edge_index)
    print(f"{bcolors.OKGREEN}{bcolors.BOLD}Computing Prototypes{bcolors.ENDC}: {perc*100.0:.2f}%", end="") 

  G = networkx.Graph(graph)
  comp = [sorted(c) for c in next(networkx.community.girvan_newman(G))]
  comp_index = {k:v for v in range(len(comp)) for k in comp[v] }

  G = networkx.Graph([i for i in G.edges if comp_index[i[0]] == comp_index[i[1]]])
  bw = networkx.betweenness_centrality(G)
  break
#%%
Edegree = [np.mean([G.degree[j] for j in comp[i]]) for i in range(len(comp))]
 
# %%
mark = {i:-1 for i in range(len(G.nodes))}

def dfs( node, level ):
  
  mark[node] = level
  if level == 3:
    return
    
  for ady in G[node]:
    if mark[ady] != -1 and mark[ady] <= mark[node]:
      continue
    dfs(ady, level + 1)
nodes = sorted([i for i in range(len(G.nodes))], reverse=True, key = lambda node : bw[node])
#%%
proto = []
for i in nodes:
  if mark[i] == -1 or mark[i] == 3:
    proto += [i]
    dfs(i, 1)
    

# %%
