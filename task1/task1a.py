# Imports 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
import time
    #! Can I use sklearn????


#! TODO
    #! Run the task code
    #! Think about regularization
    #!  Criticize the results

# Run task.py
import task.py

# Assuming the polynomial_fun, fit_polynomial_lstsq, and fit_polynomial_sgd functions are defined as in your code

# Function for cross-validation
def cross_validate(X, y, M_values, learning_rate, minibatch_size, n_splits=5):
    """
    Performs k-fold cross-validation for polynomial fitting using TensorFlow.

    Parameters:
    X (array): Input features.
    y (array): Target values.
    M_values (list): Degrees of the polynomial to evaluate.
    learning_rate (float): Learning rate for SGD.
    minibatch_size (int): Size of each minibatch for SGD.
    n_splits (int): Number of folds for cross-validation.

    Returns:
    None
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for M in M_values:
        lstsq_rmse = []
        sgd_rmse = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Least squares
            start_time = time.time()
            w_hat_lstsq = fit_polynomial_lstsq(X_train, y_train, M)
            lstsq_time = time.time() - start_time
            y_pred_lstsq = polynomial_fun(w_hat_lstsq, X_test)
            rmse_lstsq = tf.sqrt(tf.reduce_mean(tf.square(y_pred_lstsq - y_test)))
            lstsq_rmse.append(rmse_lstsq)

            # SGD
            start_time = time.time()
            w_hat_sgd = fit_polynomial_sgd(X_train, y_train, M, learning_rate, minibatch_size)
            sgd_time = time.time() - start_time
            y_pred_sgd = polynomial_fun(w_hat_sgd, X_test)
            rmse_sgd = tf.sqrt(tf.reduce_mean(tf.square(y_pred_sgd - y_test)))
            sgd_rmse.append(rmse_sgd)
        
        # Report cross-validation results
        print(f"M={M}")
        print(f"Least Squares RMSE: {tf.reduce_mean(lstsq_rmse)} ± {tf.math.reduce_std(lstsq_rmse)}, Time: {lstsq_time} seconds")
        print(f"SGD RMSE: {tf.reduce_mean(sgd_rmse)} ± {tf.math.reduce_std(sgd_rmse)}, Time: {sgd_time} seconds")
        print("----------------------------------")

# Generate synthetic data (as per your original code)
np.random.seed(0)
X = np.random.uniform(low=-20, high=20, size=30)  # Combined dataset for cross-validation
t = polynomial_fun([1, 2, 3], X) + np.random.normal(loc=0, scale=0.5, size=30)

# Cross-validation parameters
M_values = [2, 3, 4]
learning_rate = 0.01
minibatch_size = 10
n_splits = 5

# Perform cross-validation
cross_validate(X, t, M_values, learning_rate, minibatch_size, n_splits)




################################################################################
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import time

# Polynomial regression model in TensorFlow
class PolynomialModel(tf.Module):
    def __init__(self, degree):
        self.degree = degree
        # Initialize coefficients with small random values
        self.coefficients = tf.Variable(tf.random.normal([degree + 1]))

    def __call__(self, x):
        # Compute polynomial value
        powers = tf.range(self.degree + 1, dtype=tf.float32)
        terms = tf.pow(tf.expand_dims(x, axis=-1), powers)
        return tf.reduce_sum(self.coefficients * terms, axis=-1)

def compute_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def fit_with_sgd(X_train, y_train, model, learning_rate, epochs, batch_size):
    optimizer = tf.optimizers.SGD(learning_rate)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(X_train)
            loss = compute_loss(y_train, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def cross_validate_with_tf(X, y, M_values, learning_rate, epochs, batch_size, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for M in M_values:
        sgd_rmse = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Prepare the model
            model = PolynomialModel(M)

            # Fit with SGD
            start_time = time.time()
            fit_with_sgd(X_train, y_train, model, learning_rate, epochs, batch_size)
            sgd_time = time.time() - start_time

            # Validate
            y_pred_val = model(X_val)
            rmse = tf.sqrt(tf.reduce_mean(tf.square(y_pred_val - y_val)))
            sgd_rmse.append(rmse.numpy())

        # Report results
        print(f"M={M}")
        print(f"SGD RMSE: {np.mean(sgd_rmse)} ± {np.std(sgd_rmse)}, Time: {sgd_time} seconds")
        print("----------------------------------")

# Generate synthetic data
np.random.seed(0)
X = np.random.uniform(low=-20, high=20, size=30).astype(np.float32)
t = polynomial_fun([1, 2, 3], X) + np.random.normal(loc=0, scale=0.5, size=30).astype(np.float32)

# Cross-validation parameters
M_values = [2, 3, 4]
learning_rate = 0.01
epochs = 1000
batch_size = 10

# Perform cross-validation with TensorFlow
cross_validate_with_tf(X, t, M_values, learning_rate, epochs, batch_size)
