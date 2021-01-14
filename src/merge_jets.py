import numpy as np

def merge_jets(y, idx):
    size = 0
    for i in range(len(y)):
        size += len(y[i])
    y_merged = np.zeros(size)
    for i in range(len(y)):
        y_merged[idx[i]] = y[i].reshape(len(y[i]),1)
    return y_merged
