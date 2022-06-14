
#%%

import pandas as pd
data_path = '../data/augmented_train.csv'
data = pd.read_csv(data_path)
text = data['text'].to_numpy()
labels = data[['hate','irony']].astype(int).to_numpy()


#%%
import os, csv, pandas as pd
from sklearn.model_selection import train_test_split

csvfile_train = open(os.path.join(f'../data/augmented_train.csv'), 'wt', newline='', encoding="utf-8")
csvfile_test = open(os.path.join(f'../data/augmented_test.csv'), 'wt', newline='', encoding="utf-8")

spamwriter_train = csv.writer(csvfile_train, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
spamwriter_train.writerow(['text', 'hate', 'irony'])

spamwriter_test = csv.writer(csvfile_test, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
spamwriter_test.writerow(['text', 'hate', 'irony'])
 
#Hate
data = pd.read_csv('../data/hate/hateval2019_en_train.csv')
x = data['text'].to_numpy()
y = data['HS'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

for i in range(len(X_train)):
  spamwriter_train.writerow([X_train[i].replace('\n', ' '), y_train[i], -1])
for i in range(len(X_test)):
  spamwriter_test.writerow([X_test[i].replace('\n', ' '), y_test[i], -1])

data = pd.read_csv('../data/irony/SemEval2018-T3-train-taskA.txt', sep='	')

x = data['Tweet text'].to_numpy()
y = data['Label'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

for i in range(len(X_train)):
  spamwriter_train.writerow([X_train[i].replace('\n', ' '),-1,  y_train[i]])
for i in range(len(X_test)):
  spamwriter_test.writerow([X_test[i].replace('\n', ' '), -1,  y_test[i]])

# %%


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