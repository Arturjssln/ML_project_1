# Machine Learning Project 1 - Higgs boson

Project one of Machine Learning course at EPFL. This repository complete the submitted report.

## Prerequisites

Python3 with Jupyter Notebook. Numpy should be installed. Otherwise, no external libraries have been used.

## Preparation

Please first unzip train.csv and test.csv files located under "data" folder.

## Running the tests

Please run `python3 run.py` from folder `script/`

## Files

#### src/cleaners.py

Contains set of functions to prepare the data before use and features normalization.

- `normalize_features(x)`
- `prepare_data(x, replace_with, standardize, outliers, low, high, log)`
- `standardize_column(col)`


#### src/costs.py

Contains set of functions to calculate costs.

- `compute_error(y,tx,w)`
- `calculate_mse(e)`
- `calculate_rmse(e)`
- `calculate_nll(y,tx,w)`
- `calculate_reg_null(y,tx,w,lambda_)`

#### src/cross_validation.py

Contains set of functions to cross validate models.

- `build_k_indices(y, k_fold, seed)`
- `cross_validation_k_set(y, x, k_indices, k)`
- `cross_validation_k(y, x, k_indices, lambda_, degree, k)`
- `cross_validations_values(y, x, k_indices, lambda_, degree)`
- `cross_validation(y, x, k_indices, lambda_, degree)`
- `cross_validation_mean(y, x, k_indices, lambda_, degree)`
- `cross_validation_std(y, x, k_indices, lambda_, degree)`
- `cross_validation_visualization(
    lambds,
    mse_tr,
    mse_te,
    tr_label="train error",
    te_label="test error",
    show_error=False,
    std_tr=[],
    std_te=[],
    best_lambda=None,
)`
- `cross_validation_demo(y, x, degree, show_error)`
- `find_best_lambda_cv(
    y, x, degree, seed=1, k_fold=4, lambdas=np.logspace(-4, 0, 30, visualize)`
-

#### src/evaluate.py

This file contains set of functions to evaluate our model based on accuracy and F1 score.

- `compute_model_accuracy(x, y, w)`
- `calculate_f1(x, y, w)`
- `print_evaluate_model(w, y, x, loss_fn)`
- `train_fn(fn, *params):`

#### src/extend_features.py

This file contains set of functions to extend our features

- `remove_features(X, feature_names, jet_num)`
- `extend_features(X, feature_names, degree, jet_num, add_arcsinh_feature, add_log_features, add_momentum_features, remove_jet_column, equivalent)`

#### src/merge_jets.py

This file contains set of functions to merge back the differents jets.

- `merge_jets(y, idx)`

#### src/step_wise.py
This file handle step_wise process to select the bests features

- `results_r2_stepwise(list_r2_adj, indices_features)`
- `step_wise(x_cand, features, y, method, loss_method, gamma, method_minimization, threshold, maxiters, lambda_, Log)`

#### src/utils.py
Contains a set of differents helper functions

- `unzip(zipped)`
- `array_map(f, *x)`
- `arrayMap(f, *x)`
- `sigmoid(t)`
- `split_data(x, y, ratio, seed=1)`
- `build_poly(x, degree)`
- `split_indices_by_jet_num(x, index_jet_column)`
- `split_by_jet_num(x, y, index_jet_column, remove_jet_column, col_names)`
- `remove_jet_column(x, col_names)`

## Compute Lambdas

In order to compute lambdas, please open the notebook `run_v1.1_optimize_ridge.py` located in `src` folder. 
