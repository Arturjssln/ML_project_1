# -*- coding: utf-8 -*-
import numpy as np


def unzip(zipped):
    return [i for i, _ in zipped], [j for _, j in zipped]


def array_map(f, *x):
    """
    apply f function on every element of x
    """
    return np.array(list(map(f, *x)))


# @deprecated
def arrayMap(f, *x):
    return array_map(f, *x)

def sigmoid(t):
    """apply sigmoid function on t."""

    return 1/(1+np.exp(-t))

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # source: https://stackoverflow.com/a/3677283
    indices = np.random.permutation(x.shape[0])
    rate = int(np.floor(indices.shape[0] * ratio))
    training_idx, test_idx = indices[:rate], indices[rate:]
    training = (x[training_idx], y[training_idx])
    test = (x[test_idx], y[test_idx])
    return training, test


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    ones = np.ones(x.shape[0]).reshape((x.shape[0], 1))
    polys = np.concatenate([x ** (n + 1) for n in range(degree)], axis=1)
    tx = np.concatenate([ones, polys], axis=1)
    return tx


def split_indices_by_jet_num(x, index_jet_column = 22):
    """ indicates the order for merging jets """
    indices = []
    for j in range(4):
        indices.append(np.argwhere(x[:, index_jet_column] == j))
    return indices

def split_by_jet_num(x, y, index_jet_column = 22, remove_jet_column = True, col_names = None):
    """ separate data with jet_num """
    if col_names is not None:
        for i in range(len(col_names)):
            if col_names[i] == 'PRI_jet_num':
                index_jet_column = i
                break
        if remove_jet_column:
            col_names = np.delete(col_names, index_jet_column)
    JET_COUNT = 4
    jets_data = list(range(JET_COUNT))
    for j in range(JET_COUNT):
        if remove_jet_column:
            new_x = np.delete(x[x[:, index_jet_column] == j], index_jet_column, axis=1)
        else:
            new_x = x[x[:, index_jet_column] == j]
        if y is not None:
            new_y = y[x[:, index_jet_column] == j]
        else:
            new_y = None
        jets_data[j] = (new_x, new_y)
    return jets_data

def remove_jet_column(x, col_names):
    for i in range(len(col_names)):
        if col_names[i] == 'PRI_jet_num':
            index_jet_column = i
            break
    col_names = np.delete(col_names, index_jet_column)
    new_x = np.delete(x, index_jet_column, axis=1)
    return new_x, col_names
