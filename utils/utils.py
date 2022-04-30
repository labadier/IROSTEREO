import numpy as np, xml.etree.ElementTree as XT, glob
from utils.params import params
import os, re, pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torch

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Data(Dataset):

  def __init__(self, data):

    self.data = data
    
  def __len__(self):
    return len(self.data[list(self.data.keys())[0]])

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    ret = {key: self.data[key][idx] for key in self.data.keys()}
    return ret

class TokensMixed(object):
  
  def __init__(self, filepath = 'data/hurtlex_EN_conservative.tsv'):
    super(TokensMixed, self).__init__()
    self.tree = self.armarArbol(filepath)


  def armarArbol(self, filepath):
    c = None
    file = pd.read_csv(filepath, sep='\t', header=None, dtype=str)
    
    c = sorted([ (len(i), i) for i in set(file[2].to_list()) if not any(x in i for x in params.remove) ], reverse=True)
    texto = '|'.join([i[1] for i in c] )
    return re.compile(texto)

  def split(self, text, lista=False, min_leng=1):
    text = text.lower()
    text = self.tree.findall(text)
    to = []
    for t in text:
      if len(t) >= min_leng:
        to.append(t)
    text = to 
    if lista == False:
      text = ' '.join(text)
    return text
		
def setTree_fromfile():
  
	global TREE
	TREE = TokensMixed()

def hashtagWordSep(text):
	assert TREE != None
	text = text.split()
	solution = []

	for word in text:
		if word[0] == '#':
			solution.append('#')
			solution += TREE.split(word, lista=True)
		else:
			solution.append(word)
	return ' '.join(solution)


def read_truth(data_path):
    
    with open(data_path + '/truth.txt') as target_file:

        target = {}
        for line in target_file:
            inf = line.split(':::')
            target[inf[0]] = int (inf[1])  #! Change for IROSTEREO int (not 'NI' in inf[1]) and for HATER int (inf[1]) 

    return target

def removeTokens(text) -> str:
  return " ".join([i for i in text.split() if i[0] != '#' or i[-1] != '#'])
  

def load_data_PAN(data_path, labeled=True):

    addrs = sorted(np.array(glob.glob(data_path + '/*.xml')))
    setTree_fromfile()

    authors = {}
    indx = []
    label = []
    tweets = []

    if labeled == True:
        target = read_truth(data_path)

    for adr in addrs:

        author = adr[len(data_path)+1: len(adr) - 4]
        if labeled == True:
            label.append(target[author])
        authors[author] = len(tweets)
        indx.append(author)
        tweets.append([])

        tree = XT.parse(adr)
        root = tree.getroot()[0]
        for twit in root:
          tweets[-1].append(removeTokens(twit.text))
        tweets[-1] = np.array(tweets[-1])
    if labeled == True:
        return tweets, indx, np.array(label)
    return tweets, indx

def loadAugmentedData(data_path):

  data = pd.read_csv(data_path)
  text = data['text'].to_numpy()
  labels = data['label'].astype(int).to_numpy()
  m = np.random.permutation(len(labels))

  text = text[m]
  labels = labels[m]
  return text, labels

def ConverToClass(tweets, labels):

    example = []
    label = []

    for i, j in zip(tweets, labels):
        example += list(i)
        label += [j]*len(i)

    example, label = np.array(example), np.array(label)
    m = np.random.permutation(len(example))
    return example[m], label[m]

def plot_training(history, language, measure='loss'):

  plt.plot(history[measure])
  plt.plot(history['dev_' + measure])
  plt.legend(['train', 'dev'], loc='upper left')
  plt.ylabel(measure)
  plt.xlabel('Epoch')
  if measure == 'loss':
      x = np.argmin(history['dev_loss'])
  else: x = np.argmax(history['dev_acc'])

  plt.plot(x,history['dev_' + measure][x], marker="o", color="red")

  if os.path.exists('./logs') == False:
      os.system('mkdir logs')

  plt.savefig('./logs/train_history_{}.png'.format(language))

def evaluate(truthPath, dataPath, language):

  from sklearn.metrics import classification_report

  addrs = sorted(np.array(glob.glob(os.path.join(dataPath, language, '*.xml'))))
  target = read_truth(os.path.join(truthPath, language))
  y = []
  y_hat = []
  for adr in addrs:
    node = XT.parse(adr).getroot()
    y += [target[node.attrib['id']]]
    y_hat += [int(node.attrib['type'])]
  print(y_hat, y)
  print(classification_report(y, y_hat, target_names=['Negative', 'Positive'],  digits=4, zero_division=1))