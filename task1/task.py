# Stochastic Minibatch Gradient Descent for Linear Models

# Importing libraries
import numpy as np
import time
import torch


# Definiting functions

################ polynomial_fun ################

def polynomial_fun(w, x):
    """
    Calculates the value of a polynomial function at a given point.

    Parameters:
    w (torch.Tensor): Coefficients of the polynomial function.
    x (torch.Tensor): The point(s) at which to evaluate the polynomial function.

    Returns:
    torch.Tensor: The value(s) of the polynomial function at the given point(s).
    """
    M = w.shape[0]
    if x.dim() == 0:
        y = torch.sum((w[i] * torch.pow(x, i) for i in range(M)), dim=-1)
    else:
        y = torch.sum((w[i] * torch.pow(x, i) for i in range(M)), dim=-1)
    
    return y


################ fit_polynomial_lstsq ################
def fit_polynomial_lstsq(x, t, M):
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

    ## Shape
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")
    if t.ndim != 1:
        raise ValueError("t must be a 1D tensor")
    if x.shape[0] != t.shape[0]:
        raise ValueError("x and t must have the same size")

    # Fit the polynomial function to the data
    A = torch.vander(x, M + 1)
    w, _ = torch.lstsq(t.unsqueeze(1), A)
    return w


################ fit_polynomial_sgd ################
def fit_polynomial_sgd(x, t, M, learning_rate, minibatch_size, period=1000):
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
    ## Type
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

    ## Shape
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")
    if t.ndim != 1:
        raise ValueError("t must be a 1D tensor")
    if x.shape[0] != t.shape[0]:
        raise ValueError("x and t must have the same size")

    # Fit the polynomial function to the data
    ## Initialize the coefficients
    w = torch.randn(M + 1, requires_grad=True)
    ## Perform the SGD algorithm
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

#! Check the SGD on TensorFlow 
#! Make sure there's backpropag



############################################
# Questions

## Question 1: Generating a training and test set

# Define the input data
M=2
w = tf.constant([1, 2, 3], dtype=tf.float32)

# Generate training set
np.random.seed(0)
x_train = np.random.uniform(low=-20, high=20, size=20)
y_train = polynomial_fun(w, x_train) #ground truth
t_train = y_train + np.random.normal(loc=0, scale=0.5, size=20)

# Generate test set
x_test = np.random.uniform(low=-20, high=20, size=10)
y_test= polynomial_fun(w, x_test) #ground truth
t_test = y_test + np.random.normal(loc=0, scale=0.5, size=10)


## Question 2: Fitting the training set (polynomial function) using least squares

# Fit polynomial function using fit_polynomial_lstsq
M_values = [2, 3, 4]
w_hat_values = []
y_train_values = []
y_test_values = []
lstsq_fit_times= []
lstsq_train_times = []
lstsq_test_times = []

for M in M_values:
    print(f"M={M}")
    
    # Compute the optimum weight vector
    start_time = time.time()
    w_hat = fit_polynomial_lstsq(x_train, t_train, M)
    end_time = time.time()
    lstsq_fit_times.append(end_time - start_time)
    w_hat_values.append(w_hat)

    # Compute predicted target values for training set
    start_time = time.time()
    y_train = polynomial_fun(w_hat, x_train) #!TODO: Delete time
    end_time = time.time()
    lstsq_train_times.append(end_time - start_time)
    y_train_values.append(y_train)

    # Compute predicted target values for test set
    start_time = time.time()
    y_test = polynomial_fun(w_hat, x_test)
    end_time = time.time()
    lstsq_test_times.append(end_time - start_time)
    y_test_values.append(y_test)
    
    #! TODO: Meansquare
    # Reporting a
    # Calculate the mean and standard deviation of the difference between the observed training data and the true polynomial curve
    print(f"a) Difference between observed training data and true polynomial curve")
    diff_train = t_train - polynomial_fun(w, x_train)
    mean_diff_train = np.mean(diff_train)
    std_diff_train = np.std(diff_train)
    print(f"Mean difference between observed training data and true polynomial curve: {mean_diff_train}")
    print(f"Standard deviation of difference between observed training data and true polynomial curve: {std_diff_train}")

    # Reporting b
    # Calculate the mean and standard deviation of the difference between the LS-predicted values and the true polynomial curve
    print(f"b) Difference between LS-predicted values and true polynomial curve")
    diff_ls_predicted = y_train - polynomial_fun(w, x_train)
    mean_diff_ls_predicted = np.mean(diff_ls_predicted)
    std_diff_ls_predicted = np.std(diff_ls_predicted)
    print(f"Mean difference between LS-predicted values and true polynomial curve: {mean_diff_ls_predicted}")
    print(f"Standard deviation of difference between LS-predicted values and true polynomial curve: {std_diff_ls_predicted}")
    
    print("---------------------------------------------------")
    
    
# Question 3


learning_rate = 0.01
minibatch_size = 2
epochs = 1000

# Fit polynomial function using fit_polynomial_sgd
sgd_w_hat_values = []
sgd_y_train_values = []
sgd_y_test_values = []
sgd_fit_times = []
sgd_train_times = []
sgd_test_times = []

print(f"SGD EVALUATION")
for M in M_values:
    print(f"M={M}")
    
    # Fit the polynomial function using SGD
    # => sgd_w_hat
    start_time = time.time()
    sgd_w_hat = fit_polynomial_sgd(x_train, t_train, M, learning_rate=learning_rate, minibatch_size=minibatch_size)
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
    print("Difference between observed training data and true polynomial curve ")
    diff_sgd_predicted_train = sgd_y_train - polynomial_fun(w, x_train)
    mean_diff_sgd_predicted_train = np.mean(diff_sgd_predicted_train)
    std_diff_sgd_predicted_train = np.std(diff_sgd_predicted_train)
    print(f"Mean difference between SGD-predicted values and true polynomial curve for M={M}: {mean_diff_sgd_predicted_train}")
    print(f"Standard deviation of difference between SGD-predicted values and true polynomial curve for M={M}: {std_diff_sgd_predicted_train}")

    # Difference between the SGD-predicted values and the true polynomial curve for the test set
    print("Difference between the SGD-predicted values and the true polynomial curve for the test set")
    diff_sgd_predicted_test = sgd_y_test - polynomial_fun(w, x_test)
    mean_diff_sgd_predicted_test = np.mean(diff_sgd_predicted_test)
    std_diff_sgd_predicted_test = np.std(diff_sgd_predicted_test)
    print(f"Mean difference between SGD-predicted values and true polynomial curve for M={M} (test set): {mean_diff_sgd_predicted_test}")
    print(f"Standard deviation of difference between SGD-predicted values and true polynomial curve for M={M} (test set): {std_diff_sgd_predicted_test}")

    # Calculate the root-mean-square-error (RMSE) for the weight vector
    rmse_w = np.sqrt(np.mean((sgd_w_hat - w)**2))
    print(f"RMSE for weight vector (M={M}): {rmse_w}")

    # Calculate the root-mean-square-error (RMSE) for the predicted target values
    rmse_y = np.sqrt(np.mean((sgd_y_test - polynomial_fun(w, x_test))**2))
    print(f"RMSE for predicted target values (M={M}): {rmse_y}")
    
    print("---------------------------------------------------")

# Compare the speed of the two methods
print(f"SPEED COMPARISON")
print(f"Least squares")
print(f"Time spent in fitting using least squares: {np.sum(lstsq_fit_times)} seconds")
print(f"Time spent in training using least squares: {np.sum(lstsq_train_times)} seconds")
print("-----------------")
print("SGD times")
print(f"Time spent in fitting using SGD: {np.sum(sgd_fit_times)} seconds")
print(f"Time spent in training using SGD: {np.sum(sgd_train_times)} seconds") 