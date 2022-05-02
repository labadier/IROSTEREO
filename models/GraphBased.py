import os, numpy as np, torch
import torch_geometric, torch_scatter
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from utils.utils import bcolors
from models.models import seed_worker

class GCN(torch.nn.Module):

  def __init__(self, language, **kwargs):
    super(GCN, self).__init__()

    self.conv1 = torch_geometric.nn.GATv2Conv(kwargs['features_nodes'], kwargs['hidden_channels'])

    self.lin = torch.nn.Linear(kwargs['hidden_channels'], kwargs['hidden_channels']>>1)
    self.pred = torch.nn.Sequential(torch.nn.LeakyReLU(),  torch.nn.Linear( kwargs['hidden_channels']>>1, 2))
    self.best_acc = None
    self.best_acc_train = None
    self.language = language
    self.loss_criterion = torch.nn.CrossEntropyLoss()

    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, data, get_encoding=False):
    edge_index = data.edge_index.to(self.device)
    x = self.conv1(data.x.to(self.device), edge_index)
    # x = F.dropout(x, p=0.2, training=self.training)
    x = F.leaky_relu(x, 0.001)

    x = x.reshape(data.num_graphs, -1, x.shape[-1])
    x = data.prototypes.to(self.device) @ x
    
    x = self.lin(x.squeeze())
    # x = F.dropout(x, p=0.45, training=self.training)
    
    if get_encoding == True:
      return x
    return self.pred(x)

  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    torch.save(self.state_dict(), path)

  def computeLoss(self, outputs, data):
    return self.loss_criterion(outputs, data.y.to(self.device))

  def makeOptimizer(self, lr=1e-5, decay=2e-5, multiplier=1, increase=0.1):
    return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)

  


