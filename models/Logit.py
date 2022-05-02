#%%
import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.model_selection import StratifiedKFold
from utils.utils import bcolors

def predict_example(positive, negative, u, checkp, method):

    positive_aster = positive[list( np.random.choice( range(len(positive)), int(checkp*len(positive)), replace=False))]
    sn = cosine_similarity(u, positive_aster)
    y_hat = 0
    for s in sn.squeeze():
        
        negative_aster = negative[list(np.random.choice( range(len(negative)), int(checkp*len(negative)), replace=False))]
        nu = cosine_similarity(u, negative_aster).squeeze()
        y_hat_aster = np.sum(s > nu)
        y_hat = y_hat + (y_hat_aster >= len(negative_aster)/2)

    return (y_hat >= (len(positive_aster)/2))


def K_Impostor(positive, negative, unknown, checkp, method='cosine'):
    Y = np.zeros((len(unknown), ))
    
    for i in range(len(unknown)):
        ansp = predict_example(positive, negative, unknown[[i]], checkp, method)
        ansn = predict_example(negative, positive, unknown[[i]], checkp, method)
        # print(ansp, ansn)
        Y[i] = (ansp > ansn)
    # print(Y)
    return Y
