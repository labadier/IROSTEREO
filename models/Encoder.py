from transformers import AutoModel, AutoTokenizer
from utils.utils import bcolors, Data
from utils.params import params
from torch.utils.data import DataLoader
import torch, os, random, numpy as np



class MultiTaskLoss(torch.nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        
    def sigmoid(self, z ):
      return 1./(1 + torch.exp(-z))

    def forward(self, outputs, labels):

      o_t1 = self.sigmoid(outputs[:,0]) 
      loss_t1 = -(labels[:,0]*torch.log(o_t1) + (1. - labels[:,0])*torch.log(1. - o_t1))
      o_t2 = self.sigmoid(outputs[:,1]) 
      loss_t2 = -(labels[:,1]*torch.log(o_t2) + (1. - labels[:,1])*torch.log(1. - o_t2))  

      loss_t2 = (o_t2*torch.where(labels[:,1] == -1, 0, 1)).mean()
      loss_t1 = (o_t1*torch.where(labels[:,0] == -1, 0, 1)).mean()

      harmonic_mean = 2./(1./(loss_t1 + 1e-9) + 1./(loss_t2 + 1e-9))
      return harmonic_mean


def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

def HugginFaceLoad(language, weigths_source, model_name=None):

  if model_name is not None:
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, TOKENIZERS_PARALLELISM=True)
    return model, tokenizer


  prefix = 'data' if weigths_source == 'offline' else ''
  model = AutoModel.from_pretrained(os.path.join(prefix , params.models[language]))
  tokenizer = AutoTokenizer.from_pretrained(os.path.join(prefix , params.models[language]), do_lower_case=True, TOKENIZERS_PARALLELISM=True)

  return model, tokenizer

class SeqModel(torch.nn.Module):

  def __init__(self, language, **kwargs):

    super(SeqModel, self).__init__()
		
    self.mode = kwargs['mode']
    self.best_acc = None
    self.lang = language
    self.max_length = kwargs['max_length']
    self.interm_neurons = kwargs['interm_layer_size']
    self.transformer, self.tokenizer = HugginFaceLoad( language, self.mode, None if 'model_name' not in kwargs.keys() else kwargs['model_name'])
    self.intermediate = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=self.interm_neurons), torch.nn.LeakyReLU(),
                                            torch.nn.Linear(in_features=self.interm_neurons, out_features=self.interm_neurons>>1),
                                            torch.nn.LeakyReLU())
    
    self.classifier = torch.nn.Linear(in_features=self.interm_neurons>>1, out_features=2)
    
    self.loss_criterion = MultiTaskLoss() if kwargs['multitask'] == True else torch.nn.CrossEntropyLoss()
    
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, data, get_encoding=False):

    ids = self.tokenizer(data['text'], return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.device)

    X = self.transformer(**ids)[0]

    X = X[:,0]
    enc = self.intermediate(X)
    output = self.classifier(enc)
    if get_encoding == True:

      if params.models[self.lang] == "bert-base-cased":
        return X
      return enc, output
    return output 

  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Weights Loaded{bcolors.ENDC}") 

  def save(self, path):
    torch.save(self.state_dict(), path)

  def makeOptimizer(self, lr=1e-5, decay=2e-5, multiplier=1, increase=0.1):

    params = []
    for l in self.transformer.encoder.layer:

      params.append({'params':l.parameters(), 'lr':lr*multiplier}) 
      multiplier += increase

    try:
      params.append({'params':self.transformer.pooler.parameters(), 'lr':lr*multiplier})
    except:
      print(f'{bcolors.WARNING}Warning: No Pooler layer found{bcolors.ENDC}')

    params.append({'params':self.intermediate.parameters(), 'lr':lr*multiplier})
    params.append({'params':self.classifier.parameters(), 'lr':lr*multiplier})

    return torch.optim.RMSprop(params, lr=lr*multiplier, weight_decay=decay)

  def encode(self, data, batch_size, get_log=False):
    
    self.eval()    
    devloader = DataLoader(Data(data), batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)

    with torch.no_grad():
      out = None
      log = None
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache() 
        inputs = data
    
        dev_out, dev_log = self.forward(inputs, True)
        if k == 0:
          out = dev_out
          log = dev_log
        else: 
          out = torch.cat((out, dev_out), 0)
          log = torch.cat((log, dev_log), 0)

    out = out.cpu().numpy()
    log = torch.max(log, 1).indices.cpu().numpy() 
    del devloader
    if get_log: return out, log
    return out

  def computeLoss(self, outputs, data):
    return self.loss_criterion(outputs, data['labels'].to(self.device) )
