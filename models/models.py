import torch, os, random
from utils.params import params
import numpy as np, pandas as pd
from models.Encoder import SeqModel, Data, seed_worker
# from models.GraphBased import GCN
from torch.utils.data import DataLoader
from utils.utils import bcolors
from sklearn.model_selection import StratifiedKFold


class MultiTaskLoss(torch.nn.Module):
    def __init__(self, task):
        super(MultiTaskLoss, self).__init__()
        self.task = task

    def sigmoid(self, z ):
      return 1./(1 + torch.exp(-z))

    def forward(self, outputs, labels):

      o_t1 = self.sigmoid(outputs[:,0]) 
      loss_t1 = (-(labels[:,0]*torch.log(o_t1) + (1. - labels[:,0])*torch.log(1. - o_t1))).mean()
      o_t2 = torch.nn.functional.softmax(outputs[:,1:], dim=-1)
      # print()
      loss_t2 = ((-(labels[:,1:]*torch.log(o_t2)).sum(axis=-1))*torch.where(labels[:,1] == -1, 0, 1)).mean()
      return (loss_t1*(self.task != 2) + loss_t2)/2.0


def sigmoid( z ):
  return 1./(1 + torch.exp(-z))

def compute_acc(ground_truth, predictions):
  return((1.0*(torch.max(predictions, 1).indices == ground_truth)).sum()/len(ground_truth)).cpu().numpy()

def train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, split=1):
  
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
      labels = data['labels'].to(model.device)     
      
      optimizer.zero_grad()
      outputs = model(data['text'])
      loss = model.loss_criterion(outputs, labels)
   
      loss.backward()
      optimizer.step()

      # print statistics
      with torch.no_grad():
        if j == 0:
          acc = compute_acc(labels, outputs)
          running_loss = loss.item()
        else: 
          acc = (acc + compute_acc(labels, outputs))/2.0
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
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache() 
        labels = data['labels'].to(model.device) 

        dev_out = model(data['text'])
        if k == 0:
          out = dev_out
          log = labels
        else: 
          out = torch.cat((out, dev_out), 0)
          log = torch.cat((log, labels), 0)

      dev_loss = model.loss_criterion(out, log).item()
      dev_acc = compute_acc(log, out)
      eacc.append(acc)
      edev_loss.append(dev_loss)
      edev_acc.append(dev_acc) 

    band = False

    measure = dev_acc

    if model.best_acc is None or model.best_acc < measure:
      model.save(os.path.join(output, f'{model_name}.pt'))
      model.best_acc = measure
      band = True

    ep_finish_print = f' acc: {acc:.3f} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc.reshape(-1)[0]:.3f}'
    # ep_finish_print = f' acc: {acc} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc.reshape(-1)}'

    if band == True:
      print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
    else: print(last_printed + ep_finish_print)

  return {'loss': eloss, 'acc': eacc, 'dev_loss': edev_loss, 'dev_acc': edev_acc}


def train_model_CV(model_name, lang, data, splits = 5, epoches = 4, batch_size = 8, max_length = 120, 
                    interm_layer_size = 64, lr = 1e-5,  decay=2e-5, output='logs', multitask=False,
                    model_mode='offline'):

  history = []
  skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)
  
  tmplb = data['labels']
  model_params = {'mode':model_mode, 'lang':lang}
  for i, (train_index, test_index) in enumerate(skf.split(data['text'], tmplb)):  
    
    history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
    model = SeqModel(interm_layer_size, max_length, **model_params)

    trainloader = DataLoader(Data({'text':data['text'][train_index], 'label': data['labels'][train_index]}), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
    devloader = DataLoader(Data({'text':data['text'][test_index], 'label':data['labels'][test_index]}), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)

    history.append(train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, i+1))
      
    print('Training Finished Split: {}'. format(i+1))
    del trainloader
    del model
    del devloader
    break
  return history

def save_predictions(idx, y_hat, language, path):
    
    path = os.path.join(path, language)
    if os.path.isdir(path) == False: os.system(f'mkdir -p {path}')
    
    for i in range(len(idx)):
        with open(os.path.join(path, idx[i] + '.xml'), 'w') as file:
            file.write('<author id=\"{}\"\n\tlang=\"{}\"\n\ttype=\"{}\"\n/>'.format(idx[i], language, y_hat[i]))
    print(f'{bcolors.BOLD}{bcolors.OKGREEN}Predictions Done Successfully{bcolors.ENDC}')