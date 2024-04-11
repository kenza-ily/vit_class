# Importing libraries
import numpy as np
import time
import torch
import torch.nn as nn
# import functions




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


# -------- 
import torch


def fit_polynomial_sgd_M(x, t, learning_rate, minibatch_size, epochs, M_init=2):
    """
    Fits a polynomial function to a set of data points using stochastic gradient descent.

    Parameters:
    x (torch.Tensor): The x-coordinates of the data points.
    t (torch.Tensor): The target values of the data points.
    learning_rate (float): The lear    minibatch_size (int): The number of data points to use in each minibatch.
    epochs (int): The number of epochs to run the SGD algorithm.
    M_init (int): The initial value of the polynomial degree M.

    Returns:
    torch.Tensor: The coefficients of the polynomial function that best fits the data.
    int: The optimized value of the polynomial degree M.
    """

    w = torch.randn(M_init + 1, requires_grad=True)
    M = torch.tensor(float(M_init), requires_grad=True)  # Convert M_init to float
    optimizer = torch.optim.Adam([w, M], lr=learning_rate)

    for epoch in range(epochs):
        indices = torch.randperm(x.shape[0])[:minibatch_size]
        x_batch = x[indices]
        t_batch = t[indices]
        A = torch.vander(x_batch, int(M.item()) + 1)
        y = torch.mv(A, w)
        loss = torch.mean((y - t_batch) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (epochs // 10) == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    M_optimized = int(M.item())
    return w, M_optimized


# -------
# Define the input data
M = 2
w_T = torch.tensor([1, 2, 3])
w = w_T.T


# Generate training set
torch.manual_seed(0)
x_train = torch.rand(20) * 40 - 20  # random values between -20 and 20
y_train_gt = polynomial_fun(w, x_train)  # ground truth
t_train = y_train_gt + torch.randn(20) * 0.5

# Generate test set
x_test = torch.rand(10) * 40 - 20
y_test_gt = polynomial_fun(w, x_test)  # ground truth
t_test = y_test_gt + torch.randn(10) * 0.5


# --------- 
print(f"x_train: {x_train}")
print(f"y_train_gt: {y_train_gt}")
print(f"t_train: {t_train}")

print("------")


print(f"x_test: {x_test}")
print(f"y_test_gt: {y_test_gt}")
print(f"t_test: {t_test}")


# ----------

# Fit polynomial function using fit_polynomial_lstsq

## Setting
M_values = [2, 3, 4]

## Initalising
w_hat_values = []
y_train_values = []
y_test_values = []
lstsq_fit_times = []
lstsq_train_times = []
lstsq_test_times = []


# --------

# Hyperparameters
learning_rate = 1e-1
minibatch_size = 2
epochs = 1000
# Fit the true polynomial function using least squares
w_true = fit_polynomial_ls(x_train, t_train, M_values[0])

# Fit the polynomial function using fit_polynomial_sgd_M
sgd_w_hat, sgd_M_optimized = fit_polynomial_sgd_M(
    x_train, t_train, learning_rate, minibatch_size, epochs
)

# Compute predicted target values for training set using SGD
sgd_y_train = polynomial_fun(sgd_w_hat, x_train)

# Compute predicted target values for test set using SGD
sgd_y_test = polynomial_fun(sgd_w_hat, x_test)

# Difference between the SGD-predicted values and the true polynomial curve
print(f"Optimized value of M: {sgd_M_optimized}")

print("Difference between observed training data and true polynomial curve")
diff_sgd_predicted_train = sgd_y_train - polynomial_fun(w_true, x_train)
mean_diff_sgd_predicted_train = diff_sgd_predicted_train.mean()
std_diff_sgd_predicted_train = diff_sgd_predicted_train.std()
print(
    f"Mean difference between SGD-predicted values and true polynomial curve: {mean_diff_sgd_predicted_train}"
)
print(
    f"Standard deviation of difference between SGD-predicted values and true polynomial curve: {std_diff_sgd_predicted_train}"
)

# Difference between the SGD-predicted values and the true polynomial curve for the test set
print(
    "Difference between the SGD-predicted values and the true polynomial curve for the test set"
)
diff_sgd_predicted_test = sgd_y_test - polynomial_fun(w_true, x_test)
mean_diff_sgd_predicted_test = diff_sgd_predicted_test.mean()
std_diff_sgd_predicted_test = diff_sgd_predicted_test.std()
print(
    f"Mean difference between SGD-predicted values and true polynomial curve (test set): {mean_diff_sgd_predicted_test}"
)
print(
    f"Standard deviation of difference between SGD-predicted values and true polynomial curve (test set): {std_diff_sgd_predicted_test}"
)
