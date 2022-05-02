import torch, os, random
from utils.params import params
from utils.utils import Data
import numpy as np

from models.Encoder import SeqModel, seed_worker
from models.GraphBased import GCN
from torch.utils.data import DataLoader
from utils.utils import bcolors

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity


from torch_geometric.data import Data as GData
from torch_geometric.loader import DataLoader as GLoader

import networkx

MODELS = {'gcn': GCN} | {params.models[language].split('/')[-1]: SeqModel for language in params.models.keys()}

def sigmoid( z ):
  return 1./(1 + torch.exp(-z))

def compute_acc(mode, out, data, mtl=False):

  with torch.no_grad():
  
    if mtl == True:

      if mode == 'eval':
        data = {'labels':data}
      out = torch.where(sigmoid(out) > 0.5, 1, 0).cpu()
      acc_t1 = torch.where(data['labels'][:,0] == -1, 0, 1)*(out[:,0] == data['labels'][:,0])
      acc_t1 = (torch.sum(acc_t1)/torch.sum(torch.where(data['labels'][:,0] == -1, 0, 1))).item()
      
      acc_t2 = torch.where(data['labels'][:,1] == -1, 0, 1)*(out[:,1] == data['labels'][:,1])
      acc_t2 = (torch.sum(acc_t2)/torch.sum(torch.where(data['labels'][:,1] == -1, 0, 1))).item()

      harmonic_mean = 2./(1./(acc_t1 + 1e-9) + 1./(acc_t2 + 1e-9))
      return harmonic_mean

    out = out.argmax(dim=1).cpu()


    if mode == 'gcn':
      return((1.0*(out == data.y)).sum()/len(data.y)).item()
    if mode == 'eval':
      return((1.0*(out == data)).sum()/len(data)).item()
    return((1.0*(out == data['labels'])).sum()/len(data['labels'])).item()
    
def train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, split=1, mtl=False):

  eloss, eacc, edev_loss, edev_acc = [], [], [], []

  optimizer = model.makeOptimizer(lr=lr, decay=decay)
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

      outputs = model(data)
      loss = model.computeLoss(outputs, data)
   
      loss.backward()
      optimizer.step()

      # print statistics
      with torch.no_grad():
        if j == 0:
          acc = compute_acc(model_name, outputs, data, mtl)
          running_loss = loss.item()
        else: 
          acc = (acc + compute_acc(model_name, outputs, data, mtl))/2.0
          running_loss = (running_loss + loss.item())/2.0

      if (j+1)*100.0/batches - perc  >= 1 or j == batches-1:
        
        perc = (1+j)*100.0/batches
        last_printed = f'\rEpoch:{epoch+1:3d} of {epoches} step {j+1} of {batches}. {perc:.1f}% loss: {running_loss:.3f}'
        
        print(last_printed , end="")

    model.eval()
    eloss.append(running_loss)
    with torch.no_grad():
      out = None
      log = None
      
      for data in devloader:
        torch.cuda.empty_cache() 

        labels = data.y if 'gcn' in model_name else data['labels']

        dev_out = model(data)
        out = dev_out if out is None else torch.cat((out, dev_out), 0)
        log = labels if log is None else torch.cat((log, labels), 0)


      dev_loss = model.loss_criterion(out, log.to(model.device)).item()
      dev_acc = compute_acc('eval', out, log, mtl)

      eacc.append(acc)
      edev_loss.append(dev_loss)
      edev_acc.append(dev_acc) 

    band = False
    if model.best_acc is None or model.best_acc < dev_acc:
      model.save(os.path.join(output, f'{model_name}_{split}.pt'))
      model.best_acc = dev_acc
      band = True

    ep_finish_print = f' acc: {acc:.3f} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc:.3f}'

    if band == True:
      print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
    else: print(last_printed + ep_finish_print)

  return {'loss': eloss, 'acc': eacc, 'dev_loss': edev_loss, 'dev_acc': edev_acc}


def PrepareGraph(data, beta=0.97, avecPrototypes = True) -> tuple:

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
        Edegree = [np.mean([G.degree[j] for j in comp[i]]) for i in range(len(comp))]
        Prototypes = [int(np.ceil(np.log(len(comp[i]))/np.log(Edegree[i]))) + (Edegree[i] <= 2) for i in range(len(comp))]

        bw = networkx.betweenness_centrality(G)
        comp = [sorted([(node, bw[node]) for node in comp[i]], reverse=True, key = lambda x : x[1])[:Prototypes[i]] for i in range(len(comp))]
        mask[index][[node[0] for c in comp for node in c]] = 1.0
        mask[index] /= torch.sum(mask[index])
      
      mask = mask.unsqueeze(1)
      print(f"\r{bcolors.OKGREEN}{bcolors.BOLD}Computing Prototypes{bcolors.ENDC}: {perc*100.0:.2f}%") 
    
      edge_index = [torch.tensor(graph).t().contiguous() for graph in edge_index]
      return edge_index, mask, torch.tensor(data['labels'])

    edge_index = [torch.tensor(graph).t().contiguous() for graph in edge_index]
    return edge_index, torch.tensor(data['labels'])


def prepareDataLoader(model_name, data_train, data_dev = None, batch_size = None, eval=False) -> DataLoader:

  devloader = None
  if 'gcn' in model_name:
    
    encodings = torch.tensor(data_train['encodings'])
    
    edge_index, prototypes, labels = PrepareGraph(data_train)
    # # torch.save(prototypes, 'logs/train_pot.pt')
    # edge_index, labels = PrepareGraph(data_train, avecPrototypes=False)
    # prototypes = torch.load('logs/train_pot.pt')

    trainloader = GLoader([GData(x=encodings[i], y=labels[i], edge_index=edge_index[i], prototypes=prototypes[i].view(1, 1, -1)) for i in range(len(encodings))], 
          batch_size=batch_size, shuffle=(not eval), num_workers=4, worker_init_fn=seed_worker)
    
    if data_dev is not None:

      encodings = torch.tensor(data_dev['encodings'])

      edge_index, prototypes, labels = PrepareGraph(data_dev)
      # torch.save(prototypes, 'logs/test_pot.pt')
      # edge_index, labels = PrepareGraph(data_dev, avecPrototypes=False)
      # prototypes = torch.load('logs/test_pot.pt')

      devloader = GLoader([GData(x=encodings[i], y=labels[i], edge_index=edge_index[i], prototypes=prototypes[i].view(1, 1, -1)) for i in range(len(encodings))], 
            batch_size=batch_size, shuffle=(not eval), num_workers=4, worker_init_fn=seed_worker)
    return trainloader, devloader
  

  trainloader = DataLoader(Data(data_train), batch_size=batch_size, shuffle=(not eval), num_workers=4, worker_init_fn=seed_worker)
  if data_dev is not None:
    devloader = DataLoader(Data(data_dev), batch_size=batch_size, shuffle=(not eval), num_workers=4, worker_init_fn=seed_worker)
  return trainloader, devloader



def train_model_CV(model_name, lang, data, splits = 5, epoches = 4, batch_size = 8, max_length = 120, 
                    interm_layer_size = 64, lr = 1e-5,  decay=2e-5, graph_hidden_chanels = None, 
                    features_nodes = 32, output='logs', model_mode='offline'):

  history = []
  skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)
  
  model_params = {'mode':model_mode, 'hidden_channels':graph_hidden_chanels, 
                  'features_nodes':features_nodes, 'max_length': max_length, 
                  'interm_layer_size':interm_layer_size}

  for i, (train_index, test_index) in enumerate(skf.split(np.zeros_like(data['labels']), data['labels'])):  
    
    history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
    model = MODELS[model_name](language=lang, **model_params)
    
    trainloader, devloader = prepareDataLoader(model_name=model_name, data_train={key:data[key][train_index] for key in data.keys()},
                                              data_dev = {key:data[key][test_index] for key in data.keys()}, batch_size=batch_size)
    history.append(train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, i+1))
    
    print('Training Finished Split: {}'. format(i+1))
    del trainloader
    del devloader
    del model
  return history


def train_model_dev(model_name, lang, data_train, data_dev, epoches = 4, batch_size = 8, max_length = 120, 
                    interm_layer_size = 64, lr = 1e-5, decay=2e-5, graph_hidden_chanels = None, 
                    features_nodes = 32, output='logs', model_mode='offline', mtl = False):

  history = []
  
  model_params = {'mode':model_mode, 'hidden_channels':graph_hidden_chanels,
                 'features_nodes':features_nodes, 'max_length': max_length,
                 'interm_layer_size':interm_layer_size, 'multitask': mtl}

  history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
  model = MODELS[model_name](language=lang, **model_params)

  trainloader, devloader = prepareDataLoader(model_name, data_train, data_dev, batch_size)
  history.append(train_model(model_name=model_name, model=model, trainloader=trainloader, 
                  devloader=devloader, epoches=epoches, lr=lr, decay=decay, output=output, 
                  split=1, mtl=mtl))
  del trainloader
  del model
  del devloader
  return history

def save_predictions(idx, y_hat, language, path):
    
    path = os.path.join(path, language)
    if os.path.isdir(path) == False: os.system(f'mkdir -p {path}')
    
    for i in range(len(idx)):
        with open(os.path.join(path, idx[i] + '.xml'), 'w') as file:
            file.write(f'<author id=\"{idx[i]}\"\n\tlang=\"{language}\"\n\ttype=\"{["NI","I"][y_hat[i]]}\"\n/>')
    print(f'{bcolors.BOLD}{bcolors.OKGREEN}Predictions Done Successfully{bcolors.ENDC}')


def predict(model_name, data, language, splits=1, batch_size=64, interm_layer_size = 64, max_length = 120,
                    graph_hidden_chanels = None, features_nodes = 32, model_mode='offline'):

  model_params = {'mode':model_mode, 'hidden_channels':graph_hidden_chanels,
                'features_nodes':features_nodes, 'max_length': max_length,
                'interm_layer_size':interm_layer_size}

  model = MODELS[model_name](language=language, **model_params) 
  devloader,_ = prepareDataLoader(model_name=model_name, data_train=data, batch_size=batch_size, eval=True)

  model.eval()
  y_hat = 0
  
  for i in range(splits):
    print('split', i+1)
    model.load(f'logs/{model_name}_{i+1}.pt')

    with torch.no_grad():
      out = None

      for data in devloader: 
        dev_out = model(data)
        out = dev_out if out is None else torch.cat((out, dev_out), 0)

    y_hat += out.argmax(dim=1).cpu().numpy()
  return np.where(y_hat > (splits>>1), 1, 0)


def encode(model_name, data, language, data_path, splits=1, batch_size=64, interm_layer_size = 64, max_length = 120,
                    graph_hidden_chanels = None, features_nodes = 32, model_mode='offline'):

  model_params = {'mode':model_mode, 'hidden_channels':graph_hidden_chanels,
                'features_nodes':features_nodes, 'max_length': max_length,
                'interm_layer_size':interm_layer_size}

  model = MODELS[model_name](language=language, **model_params) 
  devloader,_ = prepareDataLoader(model_name=model_name, data_train={key:data[key] for key in data.keys()}, batch_size=batch_size, eval=True)

  model.eval()

  model.load(f'logs/{model_name}_1.pt')

  with torch.no_grad():
    out = None

    for data in devloader: 
      dev_out = model(data, get_encoding=True)
      out = dev_out if out is None else torch.cat((out, dev_out), 0)
  torch.save(out.cpu(), f"logs/{'train' if 'train' in data_path else 'test'}_gcnenc_{language}.pt")
  print(f"{bcolors.OKCYAN}{bcolors.BOLD}Encodings Saved Successfully{bcolors.ENDC}")
  return out.cpu().numpy()