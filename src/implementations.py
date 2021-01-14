# -*- coding: utf-8 -*-
from costs import *
from proj1_helpers import batch_iter
from utils import sigmoid


def least_squares(y, tx, loss_fn=calculate_mse):
    """Least squares normal equations."""
    # we use lstsq to solve the equation even if there're multiple complex solutions
    # usually it just returns same as .solve
    w = np.linalg.lstsq(np.dot(tx.T, tx), np.dot(tx.T, y))[0]
    loss = loss_fn(compute_error(y, tx, w))
    return w, loss


def compute_gradient(y, tx, w):
    """Gradient for GD and SGD"""
    error = compute_error(y, tx, w)
    return -1 / y.shape[0] * np.dot(tx.T, error)


def least_squares_GD(y, tx, initial_w, max_iters, gamma, loss_fn=calculate_mse):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = loss_fn(compute_error(y, tx, w))
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, loss_fn=calculate_mse):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    i = 0
    batch_size = 1
    for mini_y, mini_x in batch_iter(y, tx, batch_size, max_iters):
        gradient = compute_gradient(mini_y, mini_x, w)
        w = w - gamma * gradient
        i = i + 1
    loss = loss_fn(compute_error(y, tx, w))
    return w, loss


def ridge_regression(y, tx, lambda_, loss_fn=calculate_mse):
    """Ridge regression equations."""
    N = y.shape[0]
    D = tx.shape[1]
    a = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    b = tx.T @ y
    w = np.linalg.lstsq(a, b, rcond=None)[0]
    loss = loss_fn(compute_error(y, tx, w))
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, threshold = 1e-5, loss_fn=calculate_nll):
    """Logistic regression."""
    w = initial_w
    losses = []

    for iter in range(max_iters):
        # get loss and update w.
        pred = sigmoid(tx.dot(w))
        y = y.reshape(y.shape[0], 1)
        grad = tx.T.dot(pred - y)
        w -= np.squeeze(gamma) * grad

        loss = loss_fn(y, tx, w)

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and iter%100 == 0:
             print(f'{iter}: {np.abs(losses[-1] - losses[-2])}')
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print(f"Converged in {iter} iterations.")
            break
        if iter == max_iters-1:
            print('WARNING: The logistic regression didn\'t converge!')
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, threshold = 1e-5, loss_fn=calculate_reg_nll):
    """Regular logistic regression."""
    w = initial_w
    losses = []
    for iter in range(max_iters):
        # get loss and update w.
        pred = sigmoid(tx.dot(w))
        y = y.reshape(y.shape[0],1)
        grad = tx.T.dot(pred - y) - lambda_ * np.abs(w)
        w -= gamma * grad
        loss = loss_fn(y, tx, w, lambda_)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and iter%100 == 0:
            print(f'{iter}: {np.abs(losses[-1] - losses[-2])}')
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print("number of iteration: ", iter)
            break
        if iter == max_iters-1:
            print('WARNING: The regular logistic regression didn\'t converge')
    return w, loss


def logistic_regression_optimize(y, tx, initial_w, max_iters, gamma, threshold = 1e-5, method = 'gd', loss_fn=calculate_nll):
    w = initial_w
    losses = []

    for iter in range(max_iters):
        pred = sigmoid(tx.dot(w))

        # apply gradient descend method
        if method == 'gd':
            # get loss and update w.
            y = y.reshape(y.shape[0], 1)
            grad = tx.T.dot(pred-y)
            w -= np.squeeze(gamma) * grad

        # apply Newton method
        elif method == 'n':
            grad = tx.T.dot(pred-y)
            S = np.zeros((y.shape[0],y.shape[0]))
            for i in range(1, y.shape[0]):
                sig = sigmoid(tx[i,:].T.dot(w))
                S[i,i] = sig*(1-sig)
            # hessian
            hess = tx.T.dot(S).dot(tx)
            # w = np.linalg.solve(hess, hess.dot(w)-gamma*grad)
            w = w - np.dot(np.linalg.inv(hess), grad)

        else:
            print('ERROR: method unknown')

        loss = loss_fn(y, tx, w)

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and iter%100 == 0:
            print(f'{iter}: {np.abs(losses[-1] - losses[-2])}')
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print(f"Converged in {iter} iterations.")
            break
        if iter == max_iters-1:
            print('WARNING: The logistic regression didn\'t converge')
    return w, loss
