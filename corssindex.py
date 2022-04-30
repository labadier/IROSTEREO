#%%
import pandas as pd
from utils.utils import read_truth
import csv

ground_truth = read_truth('data/pan22-author-profiling-training-2022-03-29/en')
examples = pd.read_csv('data/saveProfilesReduced r2d2-stsb-bertweet-base-v0.tsv', sep = '\t')

with open('data/filtered_pan_data.csv', 'wt', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['text', 'label'])

    for i in range(len(examples)):
      print()
      spamwriter.writerow([examples.iloc[i]['tweet'], ground_truth[examples.iloc[i]['id']]])
        

# %%
