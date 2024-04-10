# Importing libraries
import numpy as np
import time
import torch
import torch.nn as nn
import functions




# -------- 
def fit_polynomial_sgd_M(x, t, learning_rate, minibatch_size, epochs, M_init=2):
    """
    Fits a polynomial function to a set of data points using stochastic gradient descent.

    Parameters:
    x (torch.Tensor): The x-coordinates of the data points.
    t (torch.Tensor): The target values of the data points.
    learning_rate (float): The learning rate to use for the SGD algorithm.
    minibatch_size (int): The number of data points to use in each minibatch.
    epochs (int): The number of epochs to run the SGD algorithm.
    M_init (int): The initial value of the polynomial degree M.

    Returns:
    torch.Tensor: The coefficients of the polynomial function that best fits the data.
    int: The optimized value of the polynomial degree M.

    Prints:
    float: The loss value at each 10% epoch.
    """
    # Make sure the inputs are valid
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")
    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a tensor")
    if not isinstance(learning_rate, float):
        raise TypeError("learning_rate must be a float")
    if not isinstance(minibatch_size, int):
        raise TypeError("minibatch_size must be an integer")
    if not isinstance(epochs, int):
        raise TypeError("epochs must be an integer")
    if not isinstance(M_init, int):
        raise TypeError("M_init must be an integer")

    # Fit the polynomial function to the data
    w = torch.randn(M_init + 1, requires_grad=True)
    M = torch.tensor(float(M_init), requires_grad=True)  # Convert M_init to float
    optimizer = torch.optim.SGD([w, M], lr=learning_rate)

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



# --------

# Hyperparameters
learning_rate = 0.01
minibatch_size = 2
epochs = 1000

# Fit the true polynomial function using least squares
w_true = fit_polynomial_ls(x_train, t_train, M_values[0])

# Fit the polynomial function using fit_polynomial_sgd
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



# -------
import torch

# Assuming you have x_train, t_train, x_test, and t_test defined

# Hyperparameters
learning_rate = 1e-8
minibatch_size = 2
epochs = 100
M_values = [2, 3, 4, 5]  # List of M values to try

# Fit the true polynomial function using least squares
w_true = fit_polynomial_ls(x_train, t_train, M_values[0])

best_M = None
best_w_hat = None
best_train_diff = float("inf")
best_test_diff = float("inf")

for M in M_values:
    print(f"Trying M={M}")

    # Fit the polynomial function using fit_polynomial_sgd
    sgd_w_hat = fit_polynomial_sgd(
        x_train, t_train, M, learning_rate, minibatch_size, epochs
    )

    # Compute predicted target values for training set using SGD
    sgd_y_train = polynomial_fun(sgd_w_hat, x_train)

    # Compute predicted target values for test set using SGD
    sgd_y_test = polynomial_fun(sgd_w_hat, x_test)

    # Difference between the SGD-predicted values and the true polynomial curve
    diff_sgd_predicted_train = sgd_y_train - polynomial_fun(w_true, x_train)
    mean_diff_sgd_predicted_train = diff_sgd_predicted_train.mean().item()

    diff_sgd_predicted_test = sgd_y_test - polynomial_fun(w_true, x_test)
    mean_diff_sgd_predicted_test = diff_sgd_predicted_test.mean().item()

    if (
        mean_diff_sgd_predicted_train < best_train_diff
        and mean_diff_sgd_predicted_test < best_test_diff
    ):
        best_M = M
        best_w_hat = sgd_w_hat
        best_train_diff = mean_diff_sgd_predicted_train
        best_test_diff = mean_diff_sgd_predicted_test

print(f"Optimized value of M: {best_M}")
print(
    f"Mean difference between SGD-predicted values and true polynomial curve (training set): {best_train_diff}"
)
print(
    f"Mean difference between SGD-predicted values and true polynomial curve (test set): {best_test_diff}"
)
