#%%
import numpy as np, pandas as pd
from utils.utils import *
import torch
import glob


dataPath = 'outputs'
language = 'en'
truthPath = 'data/hater/train'


evaluate(truthPath, dataPath, language)
# %%

# %%
