import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def scale_transform_normalize(coords):
    """
    Parameters:
    coords (ndarray): array of (x,y,c) coordinates

    Returns:
    ndarray: coords scaled to 1x1 with center at (0,0)
    ndarray: confidence scores of each joint
    """
    coords, scores = coords[:,:,:-1], coords[:,:,-1]
    diff = coords.max(axis=1) - coords.min(axis=1)
    diff_max = np.max(diff, axis=0)
    mean = coords.mean(axis=1).reshape(coords.shape[0],1,coords.shape[-1])
    out = (coords - mean) / diff_max
    
    return out, scores

# Read all npy files
X = []
y = []
k = 0
for f in os.scandir("./coords/openpose"):
    if f.is_file() and f.name != '.DS_Store':
        x = np.load(f)
        
        # Remove empty coords
        x = [coords for coords in x if 1 in coords.shape]
        x = np.concatenate(x)
        
        # Sanity check
        X.append(x)
        y.extend([k]*x.shape[0])
        k += 1

X = np.concatenate(X)
y = np.array(y)

# X = npy file로?

N,D,C = X.shape

# Prepare X
X_norm, scores = scale_transform_normalize(X)
scores = scores.reshape((N, D, 1))
X_norm = np.concatenate([X_norm, scores], axis=2)
X_norm = X_norm.reshape((X_norm.shape[0], 1, -1))
X_norm /= np.linalg.norm(X_norm, axis=2)[:, :, np.newaxis]

# Prepare y
y_pred = []
y_truth = []

# Grab every possible combination of 2 rows
for index in tqdm(combinations(np.arange(N), 2)):
    vec_1 = X_norm[index[0]]
    vec_2 = X_norm[index[1]]
    cosine_score = cosine_similarity(vec_1, vec_2)[0]
    y_pred.append(cosine_score)
    y_truth.append(int(y[index[0]] == y[index[1]]))