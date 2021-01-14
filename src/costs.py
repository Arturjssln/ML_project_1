# -*- coding: utf-8 -*-
import numpy as np
from utils import sigmoid

def compute_error(y, tx, w):
    return y - np.dot(tx, w)


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e ** 2)


def calculate_rmse(e):
    """Calculate the root mse for vector e."""
    mse = calculate_mse(e)
    return np.sqrt(2 * mse)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(abs(e))


def calculate_nll(y, tx, w):
    """compute the cost by negative log likelihood."""
    txw = np.dot(tx,w)
    return  np.sum(np.log(1+np.exp(txw)) - y*txw)


def calculate_reg_nll(y, tx, w, lambda_):
    """compute the cost by negative log likelihood with penalty term."""
    return np.squeeze(calculate_nll(y, tx, w) + lambda_/2.0*w.T.dot(w))
