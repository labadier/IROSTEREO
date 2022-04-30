
from calendar import EPOCH
from os import remove



class params:

  #Hate-speech-CNERG/bert-base-uncased-hatexplain
  #cardiffnlp/twitter-roberta-base-irony
  models = {'fr': 'flaubert/flaubert_base_cased', 'en': 'Hate-speech-CNERG/bert-base-uncased-hatexplain',
          'es':'finiteautomata/beto-sentiment-analysis', 'de':'oliverguhr/german-sentiment-bert',
          'it': 'dbmdz/bert-base-italian-uncased', 'pt':'neuralmind/bert-base-portuguese-cased'}

  remove = [' ', '-', '*', '+']
  LR=1e-5
  DECAY=2e-5
  ML=100
  POOL=64
  EPOCH=12
  BS=64