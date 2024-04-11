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
    optimizer = torch.optim.Adam([w], lr=learning_rate)
    for epoch in range(period):
        indices = torch.randperm(x.shape[0])[:minibatch_size]
        x_batch = x[indices]
        t_batch = t[indices]
        A = torch.vander(x_batch, M + 1)
        y = torch.mv(A, w)
        loss = torch.mean((y - t_batch) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (period // 10) == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    return w


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

## Looping
print(f"LEAST SQUARE EVALUATION")
for M in M_values:
    print(f"M={M}")

    # Compute the optimum weight vector
    start_time = time.time()
    w_hat = fit_polynomial_ls(x_train, t_train, M)
    end_time = time.time()
    lstsq_fit_times.append(end_time - start_time)
    w_hat_values.append(w_hat)

    # Compute predicted target values for training set
    start_time = time.time()
    y_train = polynomial_fun(w_hat, x_train)
    end_time = time.time()
    lstsq_train_times.append(end_time - start_time)
    y_train_values.append(y_train)

    # Compute predicted target values for test set
    start_time = time.time()
    y_test = polynomial_fun(w_hat, x_test)
    end_time = time.time()
    lstsq_test_times.append(end_time - start_time)
    y_test_values.append(y_test)

    #! TODO: Meansquare, add epochs
    # Reporting a

    # Calculate the mean and standard deviation of the difference between the observed training data and the true polynomial curve
    print(f"a) Difference between observed training data and true polynomial curve")
    diff_train = t_train - polynomial_fun(w, x_train)
    mean_diff_train = diff_train.mean()
    std_diff_train = diff_train.std()
    print(
        f"Mean difference between observed training data and true polynomial curve: {mean_diff_train}"
    )
    print(
        f"Standard deviation of difference between observed training data and true polynomial curve: {std_diff_train}"
    )

    # Reporting b
    # Calculate the mean and standard deviation of the difference between the LS-predicted values and the true polynomial curve
    print(f"b) Difference between LS-predicted values and true polynomial curve")
    diff_ls_predicted = y_train - polynomial_fun(w, x_train)
    mean_diff_ls_predicted = diff_ls_predicted.mean()
    std_diff_ls_predicted = diff_ls_predicted.std()
    print(
        f"Mean difference between LS-predicted values and true polynomial curve: {mean_diff_ls_predicted}"
    )
    print(
        f"Standard deviation of difference between LS-predicted values and true polynomial curve: {std_diff_ls_predicted}"
    )

    print("---------------------------------------------------")



# --------------
print("Question")

# Hyperparameters
M_values = [2, 3, 4]
learning_rate = 1e-1
minibatch_size = 2
epochs = 1000
# Fit the true polynomial function using least squares
w = fit_polynomial_ls(x_train, t_train, M_values[0])

# Fit polynomial function using fit_polynomial_sgd
sgd_w_hat_values = []
sgd_y_train_values = []
sgd_y_test_values = []
sgd_fit_times = []
sgd_train_times = []
sgd_test_times = []

print("SGD EVALUATION")
for M in M_values:
    print(f"M={M}")

    # Fit the polynomial function using SGD
    # => sgd_w_hat
    start_time = time.time()
    sgd_w_hat = fit_polynomial_sgd(
        x_train, t_train, M, learning_rate, minibatch_size, epochs
    )
    end_time = time.time()
    sgd_fit_times.append(end_time - start_time)
    sgd_w_hat_values.append(sgd_w_hat)

    # Compute predicted target values for training set using SGD
    # => sgd_y_train
    start_time = time.time()
    sgd_y_train = polynomial_fun(sgd_w_hat, x_train)
    end_time = time.time()
    sgd_train_times.append(end_time - start_time)
    sgd_y_train_values.append(sgd_y_train)

    # Compute predicted target values for test set using SGD
    # => sgd_y_test
    start_time = time.time()
    sgd_y_test = polynomial_fun(sgd_w_hat, x_test)
    end_time = time.time()
    sgd_test_times.append(end_time - start_time)
    sgd_y_test_values.append(sgd_y_test)

    # Difference between the SGD-predicted values and the true polynomial curve
    print("Difference between observed training data and true polynomial curve")
    diff_sgd_predicted_train = sgd_y_train - polynomial_fun(w, x_train)
    mean_diff_sgd_predicted_train = diff_sgd_predicted_train.mean()
    std_diff_sgd_predicted_train = diff_sgd_predicted_train.std()
    print(
        f"Mean difference between SGD-predicted values and true polynomial curve for M={M}: {mean_diff_sgd_predicted_train}"
    )
    print(
        f"Standard deviation of difference between SGD-predicted values and true polynomial curve for M={M}: {std_diff_sgd_predicted_train}"
    )

    # Difference between the SGD-predicted values and the true polynomial curve for the test set
    print(
        "Difference between the SGD-predicted values and the true polynomial curve for the test set"
    )
    diff_sgd_predicted_test = sgd_y_test - polynomial_fun(w, x_test)
    mean_diff_sgd_predicted_test = diff_sgd_predicted_test.mean()
    std_diff_sgd_predicted_test = diff_sgd_predicted_test.std()
    print(
        f"Mean difference between SGD-predicted values and true polynomial curve for M={M} (test set): {mean_diff_sgd_predicted_test}"
    )
    print(
        f"Standard deviation of difference between SGD-predicted values and true polynomial curve for M={M} (test set): {std_diff_sgd_predicted_test}"
    )

    print("---------------------------------------------------")



# ----------
print(f"SPEED COMPARISON")

print(f"Least squares")
print(f"Time spent in fitting using least squares: {np.sum(lstsq_fit_times)} seconds")
print(
    f"Time spent in training using least squares: {np.sum(lstsq_train_times)} seconds"
)
print("-----------------")
print("SGD times")
print(f"Time spent in fitting using SGD: {np.sum(sgd_fit_times)} seconds")
print(f"Time spent in training using SGD: {np.sum(sgd_train_times)} seconds")
