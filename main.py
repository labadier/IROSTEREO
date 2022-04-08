#%%
import argparse, sys, os, numpy as np, torch, random
from utils.params import params
from utils.utils import load_data_PAN, ConverToClass, plot_training
from utils.utils import bcolors
from models.models import train_model_CV, save_predictions
import models.GraphBased
from models.Encoder import SeqModel


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def check_params(args=None):
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-l', metavar='language', default='EN', help='Task Language')
  parser.add_argument('-mode', metavar='mode', help='task')
  parser.add_argument('-phase', metavar='phase', help='Phase')
  parser.add_argument('-output', metavar='output', help='Output Path')
  parser.add_argument('-lr', metavar='lrate', default = params.LR, type=float, help='learning rate')
  parser.add_argument('-tmode', metavar='tmode', default = 'online', help='Encoder Weights Mode')
  parser.add_argument('-decay', metavar='decay', default = params.DECAY, type=float, help='learning rate decay')
  parser.add_argument('-splits', metavar='splits', default = 5, type=int, help='spits cross validation')
  parser.add_argument('-ml', metavar='max_length', default = params.ML, type=int, help='Maximun Tweets Length')
  parser.add_argument('-interm_layer', metavar='int_layer', default = params.POOL, type=int, help='Intermediate layers neurons')
  parser.add_argument('-epoches', metavar='epoches', default=params.EPOCH, type=int, help='Trainning Epoches')
  parser.add_argument('-bs', metavar='batch_size', default=params.BS, type=int, help='Batch Size')
  parser.add_argument('-dp', metavar='data_path', help='Data Path')
  parser.add_argument('-wp', metavar='wp', help='Weight Path', default=None )
  
  return parser.parse_args(args)

if __name__ == '__main__':


  parameters = check_params(sys.argv[1:])

  language = parameters.l
  phase = parameters.phase
  learning_rate = parameters.lr
  mode_weigth = parameters.tmode
  decay = parameters.decay
  splits = parameters.splits
  max_length = parameters.ml
  interm_layer_size = parameters.interm_layer
  epoches = parameters.epoches
  batch_size = parameters.bs
  data_path = parameters.dp
  weight_path = parameters.wp
  mode = parameters.mode

  output = parameters.output
  
  if mode == 'encoder':

    if phase == 'train':
      if os.path.exists('./logs') == False:
        os.system('mkdir logs')

      text, _, labels = load_data_PAN(os.path.join(data_path, language))
      text, labels = ConverToClass(text, labels)
      
      data = {'text':text, 'labels': labels}
      history = train_model_CV(model_name=params.models[language].split('/')[-1], lang=language, data=data, splits=splits, epoches=epoches, 
                    batch_size=batch_size, max_length=max_length, interm_layer_size = interm_layer_size, lr = learning_rate,  decay=decay,
                    model_mode=mode_weigth)
      
      plot_training(history[-1], language, 'acc')
      exit(0)

    if phase == 'encode':

      '''
        Get Encodings for each author's message from the Transformer-based encoders
      '''
      text,index = load_data_PAN(os.path.join(data_path, language), labeled=False)
      model_params = {'mode':mode_weigth, 'lang':language}
      model = SeqModel(interm_layer_size, max_length, **model_params)
      model.load(os.path.join(weight_path, f"{params.models[language].split('/')[-1]}.pt"))

      encodings = [model.encode( {'text':i} , batch_size, get_log=False) for i in text]
    
      torch.save(np.array(encodings), f"logs/{'train' if 'train' in data_path else 'test'}_penc_{language}.pt")
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Encodings Saved Successfully{bcolors.ENDC}")

  if mode == 'cgn':

    if phase == 'train':
      if os.path.exists('./logs') == False:
        os.system('mkdir logs')

      _, _, labels = load_data_PAN(os.path.join(data_path, language))
      
      data = {'encodings':torch.load(f"logs/{'train' if 'train' in data_path else 'test'}_penc_{language}.pt"), 'target': labels}
      history = models.GraphBased.train_GCNN(data, language, splits = splits, epoches = epoches, batch_size = batch_size, 
                           hidden_channels = interm_layer_size, lr = learning_rate,  decay=decay)

      
      plot_training(history[-1], language, 'acc')
      exit(0)

    if phase == 'test':
      if os.path.exists('./logs') == False:
        os.system('mkdir logs')

      _, idx = load_data_PAN(os.path.join(data_path, language), labeled = False)
      
      data = {'encodings':torch.load(f"logs/{'train' if 'train' in data_path else 'test'}_penc_{language}.pt"), 'target':np.zeros((len(idx), ))}
      y_hat = models.GraphBased.predict(data, language, splits = splits, batch_size = batch_size, hidden_channels = interm_layer_size)
      save_predictions(idx, y_hat, language, output)
      exit(0)
# %%
