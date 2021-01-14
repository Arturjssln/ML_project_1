# -*- coding: utf-8 -*-
import numpy as np
import datetime

from costs import compute_error, calculate_mse
from proj1_helpers import predict_labels


def compute_model_accuracy(x, y, w):
    """ Compute the accuracy of the found model """
    y_pred = predict_labels(w, x)
    size = y.shape[0]
    false_values = np.count_nonzero(y_pred.reshape(size, 1) - y.reshape(size, 1))
    diff = false_values / size
    accuracy = 1 - diff
    return 100 * accuracy

def calculate_f1(x, y, w):
    """ Compute F1 score of the found model """
    y_sol = np.copy(y)
    y_pred = predict_labels(w, x)
    nP = 2
    nS = 2
    P = np.zeros((nP, nS))
    R = np.zeros((nP, nS))
    F1 = np.zeros((nP, nS))
    M = len(y_sol)
    F1_overall = 0
    y_sol[np.where(y_sol==-1)] = 0
    y_pred[np.where(y_pred==-1)] = 0

    for i in range(nS):
        ci = sum(y_sol == i)
        for j in range(nP):
            true_value = 0
            for m in range(M):
                if y_sol[m] == i and y_pred[m] == j:
                    true_value = true_value + 1
            kj = sum(y_pred == j)
            if kj != 0:
                P[j,i] = true_value / kj
            if ci != 0:
                R[j,i] = true_value / ci
            if R[j,i]+P[j,i] != 0:
                F1[j,i] = (2*R[j,i]*P[j,i]) / (R[j,i]+P[j,i])
        F1_overall = F1_overall + ci / M * max(F1[:,i])
    return F1_overall * 100

def print_evaluate_model(w, y, x, loss_fn=calculate_mse):
    print("---", "[test]", "---")
    print("loss:", loss_fn(compute_error(y, x, w)))
    accuracy = compute_model_accuracy(x, y, w)
    F1_score = calculate_f1(x, y, w)
    print("accuracy:", accuracy, "%")
    print("F1 score:", F1_score, "%")
    print("")


def train_fn(fn, *params):
    start_time = datetime.datetime.now()
    w, loss = fn(*params)
    end_time = datetime.datetime.now()
    print("---", "[training]", "---")
    print(
        "function:",
        fn.__name__,
        "loss:",
        loss,
        "duration",
        (end_time - start_time).total_seconds() * 1000,
        "ms",
    )
    # plot_fit(w, x, y)
    return w
