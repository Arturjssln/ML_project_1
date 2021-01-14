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
from plots import *
from evaluate import *
from cross_validation import *
from proj1_helpers import *

DATA_TRAIN_PATH = '../data/train.csv'
y, x, ids, col_names = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)
print(f'x_shape={x.shape} y_shape={y.shape}')
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test,_ = load_csv_data(DATA_TEST_PATH)
print(f'x_test_shape={tX_test.shape} NO_LABELS')

# prepare data
jets_tr = split_by_jet_num(x, y, remove_jet_column = True)
jets_te = split_by_jet_num(tX_test, y = None, remove_jet_column = True)
jets_te_indx = split_indices_by_jet_num(tX_test)
jets_nb = len(jets_tr)

for j in range(jets_nb):
    (tr_x, tr_y) = jets_tr[j]
    (te_x, _) = jets_te[j]
    
    tr_size = tr_x.shape[0]
    te_size = te_x.shape[0]

    combined_data = np.concatenate([tr_x, te_x], axis=0)
    
    assert combined_data.shape[0] == (tr_size + te_size)
    
    std_combined_data = prepare_data(
        combined_data,
        replace_with="median",
        standardize=True,
        outliers=True,
        low=1,
        high=99
    )

    jet_tr_x = std_combined_data[:tr_size, :]
    jet_te_x = std_combined_data[tr_size:, :]
    
    assert jet_tr_x.shape[0] == tr_size
    assert jet_te_x.shape[0] == te_size
    
    jets_tr[j] = (jet_tr_x, tr_y)
    jets_te[j] = (jet_te_x, None)

class Compute(Thread):
    def __init__(self, fn, log = None):
        Thread.__init__(self)
        self.log = log
        self.fn = fn
        self.output = None

    def run(self):
        if self.log is not None:
            print(self.log)
        self.output = self.fn()

K_FOLD = 5
# TRANSFORMATIONS = [
#     lambda x : np.log(np.abs(1+x)),
#     lambda x : np.cos(x),
#     lambda x : np.sin(x),
#     lambda x : np.exp(x),
#     lambda x : 1 / (1 + np.exp(-x)),
#     lambda x: np.tanh(x),
#     lambda x: np.sinc(x)
# ]

def get_accuracy_ridge(y_data, x_data, lambda_, name = None):
    accs = []
    k_indices = build_k_indices(y_data, K_FOLD)
    for k in range(K_FOLD):
        (tr_x_k, tr_y_k), (te_x_k, te_y_k) = cross_validation_k_set(y_data, x_data, k_indices, k)
        w, _ = ridge_regression(tr_y_k, tr_x_k, lambda_)
        acc = compute_model_accuracy(te_x_k, te_y_k, w)
        accs.append(acc)
    if name is not None:
        print(f'{name} => {np.mean(accs)}% +/- {3*np.std(accs)}')
    return np.mean(accs)

def compute_columns_to_remove(y_data, x_std, DEGREE, lambda_):
    NB_FEATURES = x_std.shape[1]

    useful_properties = [] # (n, degree)
    not_useful_properties = [] # (n, degree)

    for degree in range(1, DEGREE+1):
        poly_data = build_poly(x_std, degree)
        default_acc = get_accuracy_ridge(y_data, poly_data, lambda_)#'default')
        for n in range(NB_FEATURES):
            pos_n = 1 + n + NB_FEATURES*(degree-1)
            jet_x_std_poly = np.delete(poly_data, pos_n, axis=1)
            acc_without_n = get_accuracy_ridge(y_data, jet_x_std_poly, lambda_)#, f'without {n}^{degree}')
            if acc_without_n >= default_acc:
                not_useful_properties.append((n, degree))
                #print(f'{n} not useful => transform or remove?')
            if acc_without_n < default_acc:
                useful_properties.append((n, degree))
                #print(f'{n} is useful => keep')

    return not_useful_properties

lambdas = [0.0004498432668969444, 0.0003727593720314938, 0.0004498432668969444, 0.0006551285568595509]
degrees = [10, 9, 9, 9]

#threads = [Compute(lambda : compute_columns_to_remove(jets_tr[n][1], jets_tr[n][0], degrees[n], lambdas[n])) for n in range(jets_nb)]
# for thread in threads:
#     thread.start()
# for thread in threads:
#     thread.join()
# columns_to_remove = [thread.output for thread in threads]

# prepare data
print()
for j in range(jets_nb):
    (tr_x, tr_y) = jets_tr[j]
    (te_x, _) = jets_te[j]

    # remove not useful columns
    print(f'Looking for features to remove in jet #{j}...')

    NB_FEATURES = tr_x.shape[1]
    columns_to_remove = compute_columns_to_remove(tr_y, tr_x, degrees[j], lambdas[j])

    tr_x = build_poly(tr_x, degrees[j])
    te_x = build_poly(te_x, degrees[j])

    pos_to_remove = [1 + n + NB_FEATURES*(deg-1) for n, deg in columns_to_remove]

    tr_x = np.delete(tr_x, pos_to_remove, axis=1)
    te_x = np.delete(te_x, pos_to_remove, axis=1)

    print(f'{len(columns_to_remove)} features removed in jet #{j}')

    jets_tr[j] = (tr_x, tr_y)
    jets_te[j] = (te_x, None)

print()
# train models
w_values = []
for j in range(jets_nb):
    (tr_x, tr_y) = jets_tr[j]
    w, loss_tr = ridge_regression(tr_y, tr_x, lambdas[j])
    print(f'#{j} training loss = {loss_tr} with hyperparameters l={lambdas[j]} d={degrees[j]} on {tr_x.shape[0]} samples')
    w_values.append(w)

# compute model accuracy with cross-validation
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
        acc = compute_model_accuracy(te_x_k, te_y_k, w)
        accs.append(acc)
    print(f'#{j} => {np.mean(accs)}% +/- {3*np.std(accs)}')
    acc_mean += np.mean(accs) * tr_x.shape[0]
    x_sum += tr_x.shape[0]
print(f'overall => {acc_mean/x_sum}%')

# predict labels
y_preds = np.ones((tX_test.shape[0],))
for j in range(jets_nb):
    (te_x, _) = jets_te[j]
    y_pred = predict_labels(w_values[j], te_x)
    y_preds[jets_te_indx[j]] = y_pred.reshape((y_pred.shape[0], 1))

OUTPUT_PATH = '../out/output_v1.3.csv'
create_csv_submission(ids_test, y_preds, OUTPUT_PATH)

print()
print(f'>>> output file generated: {OUTPUT_PATH} <<<')