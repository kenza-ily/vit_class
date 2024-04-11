# Importing libraries
import numpy as np
import time
import torch
import torch.nn as nn


# -------- polynomial_fun
def polynomial_fun(w, x):
    """
    Calculates the value of a polynomial function at a given point.

    Parameters:
    w (torch.Tensor): Coefficients of the polynomial function.
    x (torch.Tensor): The point(s) at which to evaluate the polynomial function.

    Returns:
    torch.Tensor: The value(s) of the polynomial function at the given point(s).
    """

    # Check if w and x are tensors
    if not isinstance(w, torch.Tensor):
        raise TypeError("w must be a tensor")
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")

    M = w.shape[0]
    terms = [w[i] * torch.pow(x, i) for i in range(M)]
    y = torch.stack(terms, dim=0).sum(dim=0)
    return y


# ---------- fit_polynomial_ls
def fit_polynomial_ls(x, t, M):
    """
    Fits a polynomial function to a set of data points using least squares.

    Parameters:
    x (array or list): The x-coordinates of the data points.
    t (array or list): The target values of the data points.
    M (int): The degree of the polynomial to fit to the data.

    Returns:
    array: The coefficients of the polynomial function that best fits the data.
    """
    # Convert x and t to tensors if they are lists
    if isinstance(x, list):
        x = torch.tensor(x)
    if isinstance(t, list):
        t = torch.tensor(t)

    # Make sure the inputs are valid
    ## Type
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor or list")
    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a tensor or list")
    if not isinstance(M, int):
        raise TypeError("M must be an integer")

    # Fit the polynomial function to the data
    A = torch.vander(x, M + 1)
    w = torch.linalg.lstsq(A, t.unsqueeze(1)).solution.squeeze(1)
    return w


# --------- fit_polynomial_sgd
def fit_polynomial_sgd(x, t, M, learning_rate, minibatch_size, period=100):
    """
    Fits a polynomial function to a set of data points using stochastic gradient descent.

    Parameters:
    x (torch.Tensor): The x-coordinates of the data points.
    t (torch.Tensor): The target values of the data points.
    M (int): The degree of the polynomial to fit to the data.
    learning_rate (float): The learning rate to use for the SGD algorithm.
    minibatch_size (int): The number of data points to use in each minibatch.
    period (int): The number of epochs to run the SGD algorithm.

    Returns:
    torch.Tensor: The coefficients of the polynomial function that best fits the data.

    Prints:
    float: The loss value at each 10% epoch.
    """
    # Make sure the inputs are valid
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")
    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a tensor")
    if not isinstance(M, int):
        raise TypeError("M must be an integer")
    if not isinstance(learning_rate, float):
        raise TypeError("learning_rate must be a float")
    if not isinstance(minibatch_size, int):
        raise TypeError("minibatch_size must be an integer")

    # Fit the polynomial function to the data
    w = torch.randn(M + 1, requires_grad=True)
    for epoch in range(period):
        indices = torch.randperm(x.shape[0])[:minibatch_size]
        x_batch = x[indices]
        t_batch = t[indices]
        A = torch.vander(x_batch, M + 1)
        y = torch.mv(A, w)
        loss = torch.mean((y - t_batch) ** 2)
        loss.backward()
        with torch.no_grad():
            w -= learning_rate * w.grad
            w.grad.zero_()

        if epoch % (period // 10) == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    return w
