# Useful starting lines
import numpy as np
from threading import Thread
import sys
sys.path.append("../src/")

# custom implementations / functions
from cleaners import *
from implementations import *
from utils import *
from costs import *
from evaluate import *
from cross_validation import *
from proj1_helpers import *

# import train / test data
DATA_TRAIN_PATH = '../data/train.csv'
y, x, ids, col_names = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)
print(f'x_shape={x.shape} y_shape={y.shape}')
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test,_ = load_csv_data(DATA_TEST_PATH)
print(f'x_test_shape={tX_test.shape} NO_LABELS')

# hyperparameters from previous optimizations
lambdas = [0.0004498432668969444, 0.0003727593720314938, 0.0004498432668969444, 0.0006551285568595509]
degrees = [10, 9, 9, 9]

# PREPARE DATA
print()
print('degrees:', degrees)
print('lambdas:', lambdas)

# split data in 4 data sets using jet_num feature
jets_tr = split_by_jet_num(x, y, remove_jet_column = True)
jets_te = split_by_jet_num(tX_test, y = None, remove_jet_column = True)
jets_te_indx = split_indices_by_jet_num(tX_test)
jets_nb = len(jets_tr)

# normalize features to a N(0,1)-distribution on the whole data we have (test+train)
for j in range(jets_nb):
    (tr_x, tr_y) = jets_tr[j]
    (te_x, _) = jets_te[j]

    tr_size = tr_x.shape[0]
    te_size = te_x.shape[0]

    # merge train and test subset to prepare data
    combined_data = np.concatenate([tr_x, te_x], axis=0)

    assert combined_data.shape[0] == (tr_size + te_size)

    # Preparing data:
    # - Replace undefined values with the median
    # - Replace outliers values with the median
    # - standardize data
    std_combined_data = prepare_data(
        np.copy(combined_data),
        replace_with="median",
        standardize=True,
        outliers=True,
        low=1,
        high=99
    )

    # Extend data with a polynomial basis foudn before
    std_extended_combined_data = build_poly(
        std_combined_data,
        degrees[j]
    )

    # split train and test data subset
    jet_tr_x = std_extended_combined_data[:tr_size, :]
    jet_te_x = std_extended_combined_data[tr_size:, :]

    assert jet_tr_x.shape[0] == tr_size
    assert jet_te_x.shape[0] == te_size

    jets_tr[j] = (jet_tr_x, tr_y)
    jets_te[j] = (jet_te_x, None)

# train our 4 differents models with predefined hyperparameters
w_values = []
for j in range(jets_nb):
    (tr_x, tr_y) = jets_tr[j]
    w, loss_tr = ridge_regression(tr_y, tr_x, lambdas[j])
    #w, loss_tr = least_squares(tr_y, tr_x)
    print(f'#{j} training loss = {loss_tr} with hyperparameters l={lambdas[j]} d={degrees[j]} on {tr_x.shape[0]} samples')
    w_values.append(w)

# compute relative model accuracy with cross-validation (used as a local validation)
print()
K_FOLD = 5
acc_mean = 0
x_sum = 0
for j in range(jets_nb):
    (tr_x, tr_y) = jets_tr[j]
    k_indices = build_k_indices(tr_y, K_FOLD)
    accs = []
    for k in range(K_FOLD):
        (tr_x_k, tr_y_k), (te_x_k, te_y_k) = cross_validation_k_set(tr_y, tr_x, k_indices, k)
        w, loss_tr = ridge_regression(tr_y_k, tr_x_k, lambdas[j])
        #w, loss_tr = least_squares(tr_y_k, tr_x_k)
        acc = compute_model_accuracy(te_x_k, te_y_k, w)
        accs.append(acc)
    print(f'#{j} => {np.mean(accs)}% +/- {3*np.std(accs)}')
    acc_mean += np.mean(accs) * tr_x.shape[0]
    x_sum += tr_x.shape[0]
print(f'overall => {acc_mean/x_sum}%')

# predict y labels on test dataset
y_preds = np.ones((tX_test.shape[0],))
for j in range(jets_nb):
    (te_x, _) = jets_te[j]
    y_pred = predict_labels(w_values[j], te_x)
    y_preds[jets_te_indx[j]] = y_pred.reshape((y_pred.shape[0], 1))

# write them to output file
OUTPUT_PATH = '../out/output.csv'
create_csv_submission(ids_test, y_preds, OUTPUT_PATH)

print()
print(f'>>> output file generated: {OUTPUT_PATH} <<<')
