import numpy as np
from implementations import *

def results_r2_stepwise(list_r2_adj, indices_features):
    for i in range(len(list_r2_adj)):
        print('step', i+1, ': R2 adjusted =', list_r2_adj[i])

    print("-------------------------------------------------------")
    print("Number of features chosen:", len(indices_features))
    print("Indices of features chosen: ", indices_features)

def step_wise(x_cand, features, y, method, loss_method, gamma, method_minimization, threshold, maxiters, lambda_=0, Log=False):
    """
    Forward selection, starting with no variable and add one variable each iteration
    """
    print('Start STEPWISE')
    print('x_cand shape : {0}\t features shape : {1}'.format(len(x_cand), len(features)))
    #dataset size
    n_samples = x_cand.shape[0]
    n_features = y.shape[0]

    H = np.ones((n_samples,1))
    X = H
    k = 0

    if method == 'lr':
        initial_w = np.ones(X.shape[1])
        print('Start LR 1')
        w0, loss = logistic_regression_new(y, X, initial_w, maxiters, gamma, threshold)
        print('END LR 1')
        loglike = calculate_nll(y, X, w0)
        loglike /= n_samples

    elif method == 'ls':
        w0, loss = least_squares(y, X)
        loglike = calculate_nll(y, X, w0)
        loglike = loglike/n_samples

    elif method == 'rr':
        w0, loss = ridge_regression(y, X, lambda_)
        loglike = calculate_nll(y, X, w0)
        loglike /= n_samples

    else:
        print('Unknown method')

    R2 = 0 #definition: R2 = 1 - loglike/loglike = 1-1
    R2adj_0 = R2
    R2adj_max = R2adj_0
    idx_max = 0 #best feature index
    del(X)
    idx_features = []
    best_R2adj = []

    for j in range(n_features):

        R2_adj = []

        for i in range(x_cand.shape[1]): #increase features
            #add H offset row
            X = np.concatenate((H, x_cand[:,i].reshape(n_samples, 1)), axis=1)
            k = X.shape[1] - 1 #don't count offset column

            if method == 'lr':
                #print('Start LR {0}'.format(i))
                initial_w = np.ones(X.shape[1])
                ws, _ = logistic_regression_new(y, X, initial_w, maxiters, gamma, threshold)
                #print('End LR {0}'.format(i))

            elif method == 'ls':
                ws, _ = least_squares(y, X)

            elif method == 'rr':
                ws, _ = ridge_regression(y, X, lambda_)

            else:
                print('Unknown method')

            second_loglike = calculate_nll(y, X, ws)
            second_loglike /= n_samples
            R2 = 1-(second_loglike/loglike)

            # correction depending on the number of features and of samples
            #not sure about why
            R2_adj.append(R2 - (k/(n_samples-k-1)*(1-R2)))

        # take the best R2
        try:
            R2adj_chosen = np.max(R2_adj)
        except ValueError:  #raised if R2_adj is empty
            break

        best_R2adj.append(R2adj_chosen)
        idx_chosen = np.argmax(R2_adj)
        if R2adj_chosen > R2adj_max:
            # update
            R2adj_max = R2adj_chosen
            idx_max = idx_chosen

            # realloc of H with the regressor chosen so that X will be build with the new H and another potential candidate
            H = np.concatenate((H, x_cand[:,idx_max].reshape(n_samples,1)), axis = 1)
            x_cand = np.delete(x_cand,idx_max,1)
            if(Log == True):
                print('--------------------------------------------------------------------------------------------')
                print('Feature chosen: ', features[idx_max][1], '(index :', features[idx_max][0], ') |', ' R2adj = ', R2adj_chosen)
            idx_features.append(features[idx_max][0])

            #deleting the feature chosen in order not to have the combination with the same features
            del(features[idx_max])
            del(X)

        else:
            break

    return best_R2adj, idx_features
