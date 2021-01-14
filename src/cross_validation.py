# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from utils import build_poly, array_map, unzip
from costs import calculate_mse, compute_error
from implementations import *


def build_k_indices(y, k_fold, seed = 1):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_k_set(y, x, k_indices, k):
    # split indices
    te_indx = k_indices[k]
    tr_indx = np.setdiff1d(k_indices.flatten(), k_indices[k])
    return (x[tr_indx], y[tr_indx]), (x[te_indx], y[te_indx])


def cross_validation_k(y, x, k_indices, lambda_, degree, k):
    """return the loss of ridge regression."""
    (tr_x, tr_y), (te_x, te_y) = cross_validation_k_set(y, x, k_indices, k)
    # up to degree
    if degree is not None:
        tr_x = build_poly(tr_x, degree)
        te_x = build_poly(te_x, degree)
    # compute losses
    w, loss_tr = ridge_regression(tr_y, tr_x, lambda_)
    #w, loss_tr = reg_logistic_regression(tr_y, tr_x, lambda_, np.zeros((tr_x.shape[1],1)),10000,1e-5)
    loss_te = calculate_mse(compute_error(te_y, te_x, w))
    return loss_tr, loss_te


def cross_validations_values(y, x, k_indices, lambda_, degree):
    k_fold = len(k_indices)
    return [
        cross_validation_k(y, x, k_indices, lambda_, degree, k) for k in range(k_fold)
    ]

def cross_validation(y, x, k_indices, lambda_, degree):
    values = cross_validations_values(y, x, k_indices, lambda_, degree)
    return (np.mean(values, axis=0), np.std(values, axis=0))

def cross_validation_mean(y, x, k_indices, lambda_, degree):
    values = cross_validations_values(y, x, k_indices, lambda_, degree)
    return np.mean(values, axis=0)


def cross_validation_std(y, x, k_indices, lambda_, degree):
    values = cross_validations_values(y, x, k_indices, lambda_, degree)
    return np.std(values, axis=0)


def cross_validation_visualization(
    lambds,
    mse_tr,
    mse_te,
    tr_label="train error",
    te_label="test error",
    show_error=False,
    std_tr=[],
    std_te=[],
    best_lambda=None,
):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", label=tr_label)
    plt.semilogx(lambds, mse_te, marker=".", label=te_label)
    if show_error:
        plt.errorbar(lambds, mse_tr, yerr=std_tr)
        plt.errorbar(lambds, mse_te, yerr=std_te)
    plt.xlabel("lambda")
    plt.ylabel("rmse")

    if best_lambda is not None:
        plt.axvline(x=best_lambda)

    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def cross_validation_demo(y, x, degree, show_error=True):
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-4, 5, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # do it for all lambdas
    mean_std = [cross_validation(y, x, k_indices, l, degree) for l in lambdas]
    values, std_values = unzip(mean_std)
    # unzip cross_validation return values between training, test
    rmse_tr, rmse_te = unzip(values)
    std_tr, std_te = unzip(std_values)
    # visualize data
    cross_validation_visualization(
        lambdas,
        rmse_tr,
        rmse_te,
        "tr {}".format(degree),
        "te {}".format(degree),
        show_error,
        std_tr,
        std_te,
    )


def find_best_lambda_cv(
    y, x, degree, seed=1, k_fold=4, lambdas=np.logspace(-4, 0, 30), visualize=False
):
    """return the best lambda using ridge regression for each computed degree using cross validation."""
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # calculate mean and ste of rmse over folds for all lambdas
    mean_std = [cross_validation(y, x, k_indices, l, degree) for l in lambdas]
    mean_values, std_values = unzip(mean_std)
    # unzip cross_validation return values between training, test
    rmse_tr, rmse_te = unzip(mean_values)
    std_tr, std_te = unzip(std_values)
    # let's calculate the worst error possibility
    te_upper_bound = array_map(
        lambda rmse_std: rmse_std[0] + rmse_std[1], zip(rmse_te, std_te)
    )
    arg_max_te = np.argmin(te_upper_bound)
    best_lambda = lambdas[arg_max_te]
    if visualize:
        cross_validation_visualization(
            lambdas,
            rmse_tr,
            rmse_te,
            "tr {}".format(degree),
            "te {}".format(degree),
            True,
            std_tr,
            std_te,
            best_lambda,
        )
    return best_lambda, te_upper_bound[arg_max_te]
