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
        np.copy(combined_data),
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

def compute_parameters_for_jet(
    jet_num,
    MIN_DEGREE = 5,
    MAX_DEGREE = 13,
    K_FOLD = 5,
    logspace = np.logspace(-4, -3, 50),
    output_log = False
):
    if output_log:
        print(f'{jet_num}> Computing parameters with cross-validation on {K_FOLD} folds...')
    (jet_x, jet_y) = jets_tr[jet_num]
    jet_x_std = prepare_data(np.copy(jet_x), replace_with="median", standardize=True, outliers=True, low=1, high=99)

    # threads = [Compute(lambda : find_best_lambda_cv(
    #     jet_y,
    #     jet_x_std,
    #     degree,
    #     k_fold=K_FOLD,
    #     lambdas=logspace
    # ), f'{jet_num}> compute lambda for degree {degree}' if output_log else None) for degree in range(MIN_DEGREE, MAX_DEGREE+1)]
    # for thread in threads:
    #     thread.start()
    # for thread in threads:
    #     thread.join()
    # outputs = [thread.output for thread in threads]

    outputs = [find_best_lambda_cv(
        jet_y,
        jet_x_std,
        degree,
        k_fold=K_FOLD,
        lambdas=logspace
    ) for degree in range(MIN_DEGREE, MAX_DEGREE+1)]

    lambdas, losses_te = zip(*outputs)

    # if output_log:
    #     print(f'{jet_num}> degrees:', list(range(MIN_DEGREE, MAX_DEGREE+1)))
    #     print(f'{jet_num}> lambdas:', lambdas)
    #     print(f'{jet_num}> losses_:', losses_te)
    
    best_degree_index = np.argmin(losses_te)
    best_degree = best_degree_index + MIN_DEGREE
    if output_log:
        print(f'{jet_num}> Returning with degree={best_degree} and lambda={lambdas[best_degree_index]}')
    return best_degree, lambdas[best_degree_index]

def get_compute_parameters_fn(jet_num):
    # return lambda : compute_parameters_for_jet(
    #         jet_num,
    #         MIN_DEGREE = 1,
    #         MAX_DEGREE = 15,
    #         K_FOLD = 3,
    #         logspace = np.logspace(-4, -3, 40),
    #         output_log = True
    # )
    return lambda : compute_parameters_for_jet(
            jet_num,
            MIN_DEGREE = 1,
            MAX_DEGREE = 15,
            K_FOLD = 5,
            logspace = np.logspace(-4, 0, 30),
            output_log = True
    )

threads = [Compute(get_compute_parameters_fn(n)) for n in range(jets_nb)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
best_parameters = [thread.output for thread in threads]

# prepare data
degrees, lambdas = zip(*best_parameters)
print()
print('degrees:', degrees)
print('lambdas:', lambdas)
for j in range(jets_nb):
    (tr_x, tr_y) = jets_tr[j]
    (te_x, _) = jets_te[j]
    jets_tr[j] = (build_poly(tr_x,degrees[j]), tr_y)
    jets_te[j] = (build_poly(te_x,degrees[j]), None)

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

OUTPUT_PATH = '../out/output.csv'
create_csv_submission(ids_test, y_preds, OUTPUT_PATH)

print()
print(f'>>> output file generated: {OUTPUT_PATH} <<<')