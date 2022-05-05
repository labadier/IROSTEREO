#%%
import argparse, sys, os, numpy as np, torch, random
from sklearn import svm
from matplotlib.pyplot import axis
from utils.params import params
from utils.utils import load_data_PAN, ConverToClass, plot_training
from utils.utils import bcolors, evaluate, loadAugmentedData
from models.models import train_model_CV, train_model_dev, predict, save_predictions, encode

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from models.Encoder import SeqModel
from models.Logit import K_Impostor
from sklearn.svm import LinearSVC


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def check_params(args=None):
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-l', metavar='language', default='EN', help='Task Language')
  parser.add_argument('-mode', metavar='mode', help='task')
  parser.add_argument('-port', metavar='portion', type=float,  help='Portion for impostor')
  parser.add_argument('-model', metavar='model', help='model to encode')
  parser.add_argument('-phase', metavar='phase', help='Phase')
  parser.add_argument('-output', metavar='output', default = 'logs', help='Output Path')
  parser.add_argument('-lr', metavar='lrate', default = params.LR, type=float, help='learning rate')
  parser.add_argument('-tmode', metavar='tmode', default = 'online', help='Encoder Weights Mode')
  parser.add_argument('-decay', metavar='decay', default = params.DECAY, type=float, help='learning rate decay')
  parser.add_argument('-splits', metavar='splits', default = 5, type=int, help='spits cross validation')
  parser.add_argument('-ml', metavar='max_length', default = params.ML, type=int, help='Maximun Tweets Length')
  parser.add_argument('-interm_layer', metavar='int_layer', default = params.POOL, type=int, help='Intermediate layers neurons')
  parser.add_argument('-epoches', metavar='epoches', default=params.EPOCH, type=int, help='Trainning Epoches')
  parser.add_argument('-bs', metavar='batch_size', default=params.BS, type=int, help='Batch Size')
  parser.add_argument('-tp', metavar='train_path', help='Data path Training set')
  parser.add_argument('-dp', metavar='dev_path', help='Data path Dev set')
  parser.add_argument('-wp', metavar='wp', help='Weight Path', default=None )
  parser.add_argument('-mtl', metavar='mtl', help='Make Multitasking ?', default=None )
  
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
  train_path = parameters.tp
  dev_path = parameters.dp
  weight_path = parameters.wp
  mode = parameters.mode
  mtl = (parameters.mtl == 'mtl')
  model_name = parameters.model
  port = parameters.port

  output = parameters.output
  
  if mode == 'encoder':

    if phase == 'train':
      if os.path.exists('./logs') == False:
        os.system('mkdir logs')

      if not mtl: #! Change not because it was to train irony and hate with their respective data
        text, _, labels = load_data_PAN(os.path.join(train_path, language))
        text, labels = ConverToClass(text, labels)
        dataTrain = {'text':text, 'labels': labels}
      else:
        text, labels = loadAugmentedData(train_path)
        dataTrain = {'text':text, 'labels': labels}

      if dev_path is None:
        history = train_model_CV(model_name=params.models[language].split('/')[-1], lang=language, data=dataTrain,
                    splits=splits, epoches=epoches, batch_size=batch_size, max_length=max_length, interm_layer_size = interm_layer_size,
                    lr = learning_rate,  decay=decay, model_mode=mode_weigth)
      else:
        if not mtl:  #! Change not
          text, _, labels = load_data_PAN(os.path.join(train_path, language))
          text, labels = ConverToClass(text, labels)
        else:
          text, labels = loadAugmentedData(train_path)
        history = train_model_dev(model_name=params.models[language].split('/')[-1], lang=language, data_train=dataTrain,
                      data_dev={'text':text, 'labels': labels}, epoches=epoches, batch_size=batch_size, max_length=max_length, 
                      interm_layer_size = interm_layer_size, lr = learning_rate,  decay=decay, output=output, model_mode=mode_weigth,
                      mtl = not mtl)
      
      plot_training(history[-1], language, 'acc')
      exit(0)

    if phase == 'encode':

      '''
        Get Encodings for each author's message from the Transformer-based encoders
      '''
      text,index = load_data_PAN(os.path.join(train_path, language), labeled=False)

      model_params = {'mode':mode_weigth, 'max_length': max_length, 
                  'interm_layer_size':interm_layer_size, 'lang':language, 'multitask':mtl,
                  'model_name':model_name}

      model = SeqModel(language=language, **model_params)

      if weight_path is not None:
        if model_name is not None:
          model.load(os.path.join(weight_path, f"{model_name}_1.pt"))
        else:
          model.load(os.path.join(weight_path, f"{params.models[language].split('/')[-1]}_1.pt"))
      else: print(f"{bcolors.WARNING}{bcolors.BOLD}No Weights Loaded{bcolors.ENDC}")

      encodings = [model.encode( {'text':i} , batch_size, get_log=False) for i in text]
    
      torch.save(np.array(encodings), f"logs/{'train' if 'train' in train_path else 'test'}_penc_{language}.pt")
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Encodings Saved Successfully{bcolors.ENDC}")

  if mode == 'gcn':

    if phase == 'train':
      if os.path.exists('./logs') == False:
        os.system('mkdir logs')

      _, _, labels = load_data_PAN(os.path.join(train_path, language))
      dataTrain = {'encodings':torch.load(f"logs/irony/train_penc_{language}.pt")+ torch.load(f"logs/hate/train_penc_{language}.pt"),
                    'sem_encodings': torch.load(f"logs/raw/train_penc_{language}.pt"),
                    'labels': labels}
      
      if dev_path is None:
        history = train_model_CV(model_name='gcn', lang=language, data=dataTrain, splits=splits, epoches=epoches, batch_size=batch_size, 
                                  max_length=max_length, graph_hidden_chanels = interm_layer_size, lr = learning_rate,  decay=decay, model_mode=mode_weigth)
      else:
        _, _, labels = load_data_PAN(os.path.join(dev_path, language))

        dataDev = {'encodings':torch.load(f"logs/irony/test_penc_{language}.pt")+ torch.load(f"logs/hate/test_penc_{language}.pt"),
                  'sem_encodings': torch.load(f"logs/raw/test_penc_{language}.pt"),
                  'labels': labels}

        history = train_model_dev(model_name='gcn', lang=language, data_train=dataTrain, data_dev=dataDev, epoches=epoches,
                                batch_size=batch_size, max_length=max_length, graph_hidden_chanels = interm_layer_size,
                                lr = learning_rate,  decay=decay, model_mode=mode_weigth)

      
      plot_training(history[-1], language, 'acc')
      exit(0)

    if phase == 'test':
      if os.path.exists('./logs') == False:
        os.system('mkdir logs')

      _, idx = load_data_PAN(os.path.join(dev_path, language), labeled = False)
      data = {'encodings':torch.load(f"logs/irony/{'train' if 'train' in dev_path else 'test'}_penc_{language}.pt") +\
              torch.load(f"logs/hate/{'train' if 'train' in dev_path else 'test'}_penc_{language}.pt"),
              'sem_encodings': torch.load(f"logs/raw/{'train' if 'train' in dev_path else 'test'}_penc_{language}.pt"),
              'labels':np.zeros((len(idx), ))}
      y_hat = predict(model_name='gcn', data=data, language=language, splits = splits, batch_size = batch_size,
                      graph_hidden_chanels = interm_layer_size)

      save_predictions(idx, y_hat, language, output)
      exit(0)

    if phase == 'encode':
      if os.path.exists('./logs') == False:
        os.system('mkdir logs')

      _, idx = load_data_PAN(os.path.join(dev_path, language), labeled = False)
      
      data = {'encodings':torch.load(f"logs/irony/{'train' if 'train' in dev_path else 'test'}_penc_{language}.pt") +\
              torch.load(f"logs/hate/{'train' if 'train' in dev_path else 'test'}_penc_{language}.pt"),
              'sem_encodings': torch.load(f"logs/raw/{'train' if 'train' in dev_path else 'test'}_penc_{language}.pt"),
              'labels':np.zeros((len(idx), ))}

      encode = encode(model_name='gcn', data=data, language=language, data_path=dev_path, splits = splits, batch_size = batch_size,
                      graph_hidden_chanels = interm_layer_size)
      exit(0)

  if mode == "impostor":

    ''' 
      Classify the profiles with Impostors Method 
    '''

    _, _, labels = load_data_PAN(os.path.join(train_path, language.lower()), labeled=True)
    _, idx  = load_data_PAN(os.path.join(dev_path, language.lower()), labeled=False)

    encodings = torch.load('logs/train_gcnenc_en.pt')
    encodings_test = torch.load('logs/test_gcnenc_en.pt')    
      
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)   
    overl_acc = 0

    Y_Test = np.zeros((len(encodings_test),))
    for i, (train_index, test_index) in enumerate(skf.split(encodings, labels)):
      unk = encodings[test_index]
      unk_labels = labels[test_index] 

      known = encodings[train_index]
      known_labels = labels[train_index]

      y_hat = K_Impostor(positive = known[list(np.argwhere(known_labels==1).reshape(-1))], 
                         negative = known[list(np.argwhere(known_labels==0).reshape(-1))], 
                         unknown = unk, checkp=port)
      Y_Test += K_Impostor(encodings[list(np.argwhere(labels==1).reshape(-1))], 
                         encodings[list(np.argwhere(labels==0).reshape(-1))], 
                         unknown = encodings_test, checkp=port)
      
      metrics = classification_report(unk_labels, y_hat, target_names=['No Hate', 'Hate'],  digits=4, zero_division=1)
      acc = accuracy_score(unk_labels, y_hat)
      overl_acc += acc
      print('Report Split: {} - acc: {}{}'.format(i+1, np.round(acc, decimals=2), '\n'))
      print(metrics)

    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Overall Accuracy for {port} in {language}: {np.round(overl_acc/splits, decimals=2)}{bcolors.ENDC}")
    save_predictions(idx, np.where(Y_Test > (splits>>1), 1, 0), language, output)

  if mode == "svm":

    ''' 
      Classify the profiles with Impostors Method 
    '''

    _, _, labels = load_data_PAN(os.path.join(train_path, language.lower()), labeled=True)
    _, idx  = load_data_PAN(os.path.join(dev_path, language.lower()), labeled=False)

    encodings = torch.load('logs/train_gcnenc_en.pt')
    encodings_test = torch.load('logs/test_gcnenc_en.pt')    
      
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)   
    overl_acc = 0

    Y_Test = np.zeros((len(encodings_test),))
    for i, (train_index, test_index) in enumerate(skf.split(encodings, labels)):
      
      svm_model_linear = LinearSVC( max_iter =1e6 ).fit(encodings[train_index], labels[train_index])
      print(f'train svm split {i+1}: {svm_model_linear.score(encodings[train_index], labels[train_index])}')
      
      svm_predictions = svm_model_linear.predict(encodings_test)
      Y_Test += svm_predictions
      acc = svm_model_linear.score(encodings[train_index], labels[train_index])
      overl_acc += acc
      print('Report Split: {} - acc: {}{}'.format(i+1, np.round(acc, decimals=2), '\n'))

    print(f'Accuracy {language}: {np.round(overl_acc/splits, decimals=2)}')
    save_predictions(idx, np.where(Y_Test > (splits>>1), 1, 0), language, output)

  if mode == "eval":
    
    evaluate(truthPath=dev_path, dataPath='outputs', language=language)

# %%
