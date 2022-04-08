from cProfile import label
import os, numpy as np, torch
from sklearn.metrics.pairwise import cosine_similarity
import torch_geometric
from torch_geometric.data import Data as GData
from torch_geometric.loader import DataLoader as GLoader
from torch_scatter import scatter_mean
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from utils.utils import bcolors
from models.models import seed_worker

class GCN(torch.nn.Module):

	def __init__(self, language, hidden_channels=64, features_nodes=96):
		super(GCN, self).__init__()

		self.conv1 = torch_geometric.nn.GATv2Conv(features_nodes, hidden_channels)
    
		self.lin = torch.nn.Linear(hidden_channels, 64)
		self.pred = torch.nn.Sequential(torch.nn.LeakyReLU(),  torch.nn.Linear(64, 32), torch.nn.Linear(32, 2))
		self.best_acc = None
		self.best_acc_train = None
		self.language = language
		self.loss_criterion = torch.nn.CrossEntropyLoss()

		self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
		self.to(device=self.device)

	def forward(self, x, edge_index, batch, phase='train'):

		edge_index = edge_index.to(self.device)
		x = self.conv1(x.to(self.device), edge_index)
		x = F.dropout(x, p=0.2, training=self.training)
		x = x.relu()
    
		x = scatter_mean(x, batch.to(self.device), dim=0)

		x = self.lin(x)
		if phase == 'encode':
			return x
		return self.pred(x)

	def load(self, path):
		self.load_state_dict(torch.load(path, map_location=self.device))

	def save(self, path):
		torch.save(self.state_dict(), os.path.join('logs', path))
	
	def get_encodings(self, encodings, batch_size):

		self.eval()    
		a = []
		b = []
		for i in range(encodings.shape[1]):  
			a += [i]*int(encodings.shape[1])
			b += [j for j in range(encodings.shape[1])]

		edges = [a, b]
		edges = torch.tensor(edges, dtype=torch.long)
		encodings = torch.tensor(encodings, dtype=torch.float)
		data_test = [torch_geometric.data.Data(x=encodings[i], edge_index=edges) for i in range(encodings.shape[0])]
		devloader = torch_geometric.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)
		

		with torch.no_grad():
			out = None
			log = None
			for k, data in enumerate(devloader, 0):
				torch.cuda.empty_cache() 
				dev_out = self.forward(data.x, data.edge_index, data.batch, phase='encode')
				if k == 0:
					out = dev_out
				else: 
					out = torch.cat((out, dev_out), 0)

		out = out.cpu().numpy()
		del devloader
		return out


def PrepareGraph(data, beta=0.8) -> list:
  
  alpha = [cosine_similarity(i) for i in data['encodings']]
  edge_index = [list(zip(*np.where(graphs>beta))) for graphs in alpha]
  edge_index = [torch.tensor(graph).t().contiguous() for graph in edge_index]
  return edge_index, torch.tensor(data['encodings']), torch.tensor(data['target'])


def train_GCNN(data, language, splits = 5, epoches = 4, batch_size = 64, hidden_channels = 64, lr = 1e-5,  decay=2e-5):

  skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)

  edge_index, encodings, labels = PrepareGraph(data)
  overall_acc = 0

  history = []
  for i, (train_index, test_index) in enumerate(skf.split(np.zeros_like(labels), labels)):  

    history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
    model = GCN(language, hidden_channels, encodings.shape[-1])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainloader = GLoader([GData(x=encodings[i], y=labels[i], edge_index=edge_index[i]) for i in train_index], 
          batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
    devloader = GLoader([GData(x=encodings[i], y=labels[i], edge_index=edge_index[i]) for i in test_index], 
          batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)

    batches = len(trainloader)

    for epoch in range(epoches):

      running_loss = 0.0
      perc = 0
      acc = 0

      model.train()
      last_printed = ''
      for j, data in enumerate(trainloader, 0):

        torch.cuda.empty_cache()         

        optimizer.zero_grad()
        
        outputs = model(data.x, data.edge_index, data.batch)
        loss = model.loss_criterion(outputs, data.y.to(model.device))

        loss.backward()
        optimizer.step()

        # print statistics
        outputs = outputs.argmax(dim=1).cpu()
        with torch.no_grad():
          if j == 0:
            acc = ((1.0*(outputs == data.y)).sum()/len(data.y)).numpy()
            running_loss = loss.item()
          else: 
            acc = (acc + ((1.0*(outputs == data.y)).sum()/len(data.y)).numpy())/2.0
            running_loss = (running_loss + loss.item())/2.0

          if (j+1)*100.0/batches - perc  >= 1 or j == batches-1:
            perc = (1+j)*100.0/batches
            last_printed = f'\rEpoch:{epoch+1:3d} of {epoches} step {j+1} of {batches}. {perc:.1f}% loss: {running_loss:.3f}'
            print(last_printed, end="")

      model.eval()
      history[-1]['loss'].append(running_loss)
      with torch.no_grad():
        out = None
        log = None
        for k, data in enumerate(devloader, 0):
          torch.cuda.empty_cache() 

          dev_out = model(data.x, data.edge_index, data.batch)
          if k == 0:
            out = dev_out
            log = data.y
          else: 
            out = torch.cat((out, dev_out), 0)
            log = torch.cat((log, data.y), 0)

        dev_loss = model.loss_criterion(out, log.to(model.device)).item()
        out = out.argmax(dim=1).cpu()
        dev_acc = ((1.0*(out == log)).sum()/len(log)).cpu().numpy() 
          
        history[-1]['acc'].append(acc)
        history[-1]['dev_loss'].append(dev_loss)
        history[-1]['dev_acc'].append(dev_acc) 

      band = False
      if model.best_acc is None or model.best_acc < dev_acc:
        model.save('gcn_{}_{}.pt'.format(language[:2], i+1))
        model.best_acc = dev_acc
        model.best_acc_train = acc
        band = True
      elif model.best_acc == dev_acc and (model.best_acc_train is None or model.best_acc_train < acc):
        model.save('gcn_{}_{}.pt'.format(language[:2], i+1))
        model.best_acc_train = acc
        band = True

      ep_finish_print = f' acc: {acc:.3f} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc.reshape(-1)[0]:.3f}'

      if band == True:
        print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
      else: print(ep_finish_print)

    overall_acc += model.best_acc
    print('Training Finished Split: {}'. format(i+1))
    del trainloader
    del model
    del devloader
    # break
  print(f"{bcolors.OKGREEN}{bcolors.BOLD}{50*'*'}\nOveral Accuracy {language}: {overall_acc/splits}\n{50*'*'}{bcolors.ENDC}")
  return history

def predict(data, language, splits, batch_size, hidden_channels):

  edge_index, encodings, labels = PrepareGraph(data)
  model = GCN(language, hidden_channels, encodings.shape[-1]) 
  devloader = GLoader([GData(x=encodings[i], y=labels[i], edge_index=edge_index[i]) for i in range(len(labels))], 
          batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)

  model.eval()
  y_hat = 0
  for i in range(splits):

    model.load(f'logs/gcn_{language}_{i+1}.pt')
    with torch.no_grad():
      out = None
      for k, data in enumerate(devloader, 0): 

        dev_out = model(data.x, data.edge_index, data.batch)
        out = dev_out if k == 0 else torch.cat((out, dev_out), 0)

    y_hat += torch.argmax(torch.nn.functional.softmax(out, dim=-1), axis=-1).cpu().numpy()
  return np.where(y_hat > 2, 1, 0)
  


