""" Syntethic data generated linear regression with added noise to simulate real data"""
import time

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)  #function to seed the data
X_features = rng.normal(size=(100, 3)) 
#seeding the data around `Bell curve` woth 100 rows(training examples) and 3 columns(features)

X = np.hstack([np.ones((100, 1)), X_features]) 
#adding intercept as a very first column, to preserve base case, in case if features are 0
#print(X.shape)

theta_true = np.array([5.0, 2.0, -3.0, 0.5]) #actual parameters
y = X @ theta_true + 0.1 * rng.normal(size=100) #target variable within noise
#print(y.shape)

def compute_cost(X, y, theta):
    """ Iterative Cost function to find how model's prediction is actually correct

    Args:
        X: (m, n) matrix, first column for the intercept.
        y: (m,) target values.
        theta: (n,) parameters.

    Return:
        Scalar cost J(theta)
    """
    cost = 0 
    for i in range (X.shape[0]):
        prediction = X[i] @ theta 
    #dot multiplication iterated over every row and multipling elementwise and sum into one number 
        error = (prediction - y[i]) ** 2
        cost += error
    return 1/2 * cost 

print("Actual cost: ", compute_cost(X, y, theta_true))

def compute_cost_vect(X, y, theta):
    """ Vecotrized Cost function to find how model's prediction is actually correct
    
        Args:
            X: (m, n) matrix, first column for the intercept.
            y: (m,) target values.
            theta: (n,) parameters.
    
        Return:
            Scalar cost J(theta)
        """
    error = X @ theta - y 
    #actual predictions for all examples at once minus target -> error vector (100,)
    cost = 1/2 * error @ error # z.T*z = sum of z^2, @ does that sum
    return cost 

print("Actual cost using matrix form: ", compute_cost_vect(X, y, theta_true))

# the two cost implementations agree
assert np.isclose(compute_cost_vect(X, y, theta_true), compute_cost(X, y, theta_true))

def LMS(X, y, theta, alpha, num_iters):
    """ Iterative Batch Gradient Descent funtion to update parameters to minimize the cost(error)
        Args:
            X: (m, n) design matrix, first column all ones for the intercept.
            y: (m,) target values.
            theta: (n,) initial parameters.
            alpha: learning rate — step size.
            num_iters: number of full passes over the data.

        Returns:
            theta: (n,) fitted parameters.
            cost_history: list of J(theta) after each iteration."""
    
    m, n = X.shape
    cost_history = []
    for _ in range(num_iters):
        gradient = np.zeros(n)
        for j in range(n):
            for i in range(m):
                gradient[j] += (X[i] @ theta - y[i]) * X[i, j]
        theta = theta - alpha * gradient
        cost_history.append(compute_cost_vect(X, y, theta))
        #cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

theta_init = np.zeros(X.shape[1])
start = time.perf_counter()
theta_gd, cost_history = LMS(X, y, theta_init, alpha=0.01, num_iters=500)

print("Cost using BGD: ", cost_history[-1])

print(f"LMS (loops):      {time.perf_counter() - start:.3f}s")

def LMS_vector(X, y, theta, alpha, num_iters):
    """Vectorized Batch Gradient Descent funtion to update parameters to minimize the cost(error)
        Args:
            X: (m, n) design matrix, first column all ones for the intercept.
            y: (m,) target values.
            theta: (n,) initial parameters.
            alpha: learning rate — step size.
            num_iters: number of full passes over the data.

        Returns:
            theta: (n,) fitted parameters.
            cost_history: list of J(theta) after each iteration."""
    
    cost_history = []
    for _ in range(num_iters): 
        gradient = X.T @ (X @ theta - y)
        theta = theta - alpha * gradient
        cost_history.append(compute_cost_vect(X, y, theta))
        #cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

theta_init = np.zeros(X.shape[1])
start = time.perf_counter()
theta_lv, cost_history_lv = LMS_vector(X, y, theta_init, alpha=0.01, num_iters=500)

print("Cost using BGD_vector: ", cost_history_lv[-1])

print(f"LMS (vector):      {time.perf_counter() - start:.3f}s")

# loop and vectorised GD are the same algorithm
assert np.allclose(theta_lv, theta_gd)

def SGD(X, y, theta, alpha, num_epochs):
    """Stochastic Gradient Descent function updates parameters after each individual example, 
    rather than accumulating over the full dataset
        Args:
            X: (m, n) design matrix, first column all ones for the intercept.
            y: (m,) target values.
            theta: (n,) initial parameters.
            alpha: learning rate — step size.
            num_epoch: full pass, containing updates

        Returns:
            theta: (n,) fitted parameters.
            cost_history: list of J(theta) after each epoch."""
        
    m, n = X.shape
    cost_history = []
    for _ in range(num_epochs):
        for i in range(m):
            gradient = (X[i] @ theta - y[i]) * X[i]
            theta = theta - alpha * gradient 
        cost_history.append(compute_cost_vect(X, y, theta))
        #cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

theta_init = np.zeros(X.shape[1])
theta_sgd, cost_history_sgd = SGD(X, y, theta_init, alpha=0.01, num_epochs=300)

print("Cost using SGD: ", cost_history_sgd[-1])

def normal_equation(X, y):
    """Closed-form solution: theta = (X^T X)^-1 X^T y.

    Solves for the minimising theta directly by setting the gradient to zero,
    rather than iterating. No learning rate, no initial guess, no convergence.
    Only available because J is quadratic in theta(bowl shape form) — logistic regression and
    neural networks have no closed form.

    Args:
        X: (m, n) design matrix, first column all ones for the intercept.
        y: (m,) target values.

    Returns:
        (n,) fitted parameters."""
     
    return np.linalg.solve(X.T @ X, X.T @ y)

theta_ne = normal_equation(X, y)

print("Cost Using normal equation: ", compute_cost_vect(X, y, theta_ne))

# iterative and closed-form agree — the independent cross-check
assert np.allclose(theta_gd, theta_ne, atol=1e-2)
#SGD with oscilation
assert np.allclose(theta_sgd, theta_ne, atol=1e-1)
# both recover the parameters that generated the data
assert np.allclose(theta_ne, theta_true, atol=0.1)

print("GD:    ", theta_gd)
print("GD_vector:    ", theta_lv)
print("SGD:   ", theta_sgd)
print("Normal:", theta_ne)
print("True:  ", theta_true)

k = 30
plt.plot(cost_history[:k], label="Batch GD")
plt.plot(cost_history_lv[:k], label="VGD", linestyle="--")
plt.plot(cost_history_sgd[:k], label="SGD")
plt.xlabel("Iteration")
plt.ylabel("Cost J(θ)")
plt.yscale("log")
plt.legend()
plt.savefig("lectures/lec02/cost_history.png", dpi=150)
plt.show()