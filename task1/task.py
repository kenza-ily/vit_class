# Importing libraries
import numpy as np
import time
import torch
import torch.nn as nn
import functions


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
learning_rate = 1e-2
minibatch_size = 2
epochs = 100
# ! TODO add optimiser with Adam, PyTorch loss -> MSE?

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
