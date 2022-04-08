from transformers import AutoModel, AutoTokenizer
from utils.utils import bcolors
from utils.params import params
from torch.utils.data import DataLoader, Dataset
import torch, os, random, numpy as np

def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

class Data(Dataset):

  def __init__(self, data):

    self.data = {i:data[i] for i in data.keys() if i != 'label'}
    self.label = data['label'] if 'label' in data.keys() else None

  def __len__(self):
    for i in self.data:
      return len(self.data[i])

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    ret = {i:self.data[i][idx] for i in self.data.keys() if i != 'label'}
    if self.label is not None:
      ret['labels'] = self.label[idx]
    return ret

def HugginFaceLoad(language, weigths_source):

  prefix = 'data' if weigths_source == 'offline' else ''
  model = AutoModel.from_pretrained(os.path.join(prefix , params.models[language]))
  tokenizer = AutoTokenizer.from_pretrained(os.path.join(prefix , params.models[language]), do_lower_case=True, TOKENIZERS_PARALLELISM=True)

  return model, tokenizer

class SeqModel(torch.nn.Module):

  def __init__(self, interm_size, max_length, **kwargs):

    super(SeqModel, self).__init__()
		
    self.mode = kwargs['mode']
    self.best_acc = None
    self.lang = kwargs['lang']
    self.max_length = max_length
    self.interm_neurons = interm_size
    self.transformer, self.tokenizer = HugginFaceLoad( kwargs['lang'], self.mode)
    self.intermediate = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=self.interm_neurons), torch.nn.LeakyReLU(),
                                            torch.nn.Linear(in_features=self.interm_neurons, out_features=self.interm_neurons>>1),
                                            torch.nn.LeakyReLU())
    
    self.classifier = torch.nn.Linear(in_features=self.interm_neurons>>1, out_features=2)
    self.loss_criterion = torch.nn.CrossEntropyLoss()
    
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, data, get_encoding=False):

    ids = self.tokenizer(data, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.device)

    X = self.transformer(**ids)[0]

    X = X[:,0]
    enc = self.intermediate(X)
    output = self.classifier(enc)
    if get_encoding == True:
      return enc, output
    return output 

  def load(self, path):
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Weights Loaded{bcolors.ENDC}") 
    self.load_state_dict(torch.load(path, map_location=self.device))

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
        inputs = data['text']
    
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
