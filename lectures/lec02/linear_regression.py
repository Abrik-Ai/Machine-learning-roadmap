import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
X_features = rng.normal(size=(100, 3))

X = np.hstack([np.ones((100, 1)), X_features])
#print(X.shape)

theta_true = np.array([5.0, 2.0, -3.0, 0.5])
y = X @ theta_true + 0.1 * rng.normal(size=100)
#print(y.shape)

def compute_cost(X, y, theta):
    cost = 0 
    for i in range (X.shape[0]):
        prediction = X[i] @ theta
        error = (prediction - y[i]) ** 2
        cost += error
    return 1/2 * cost 

print("Actual cost: ", compute_cost(X, y, theta_true))
def compute_cost_vect(X, y, theta):
    error = X @ theta - y
    cost = 1/2 * error @ error
    return cost 

print("Actual cost using matrix form: ", compute_cost_vect(X, y, theta_true))
#print(compute_cost(X, y, theta_true))
#print(compute_cost_vect(X, y, theta_true))

assert np.isclose(compute_cost_vect(X, y, theta_true), compute_cost(X, y, theta_true))

def LMS(X, y, theta, alpha, num_iters):
    m, n = X.shape
    cost_history = []
    for _ in range(num_iters):
        gradient = np.zeros(n)
        for j in range(n):
            for i in range(m):
                gradient[j] += (X[i] @ theta - y[i]) * X[i, j]
        theta = theta - alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

theta_init = np.zeros(X.shape[1])
theta_gd, cost_history = LMS(X, y, theta_init, alpha=0.01, num_iters=500)

print("Cost using BGD: ", cost_history[-1])

def SGD(X, y, theta, alpha, num_epochs):
    m, n = X.shape
    cost_history = []
    for _ in range(num_epochs):
        for i in range(m):
            gradient = (X[i] @ theta - y[i]) * X[i]
            theta = theta - alpha * gradient 
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

theta_init = np.zeros(X.shape[1])
theta_sgd, cost_history_sgd = SGD(X, y, theta_init, alpha=0.01, num_epochs=300)

print("Cost using SGD: ", cost_history_sgd[-1])

def normal_equation(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

theta_ne = normal_equation(X, y)
print("GD:    ", theta_gd)
print("SGD:   ", theta_sgd)
print("Normal:", theta_ne)
print("True:  ", theta_true)

assert np.allclose(theta_gd, theta_ne, atol=1e-2)
assert np.allclose(theta_ne, theta_true, atol=0.1)
assert np.allclose(theta_sgd, theta_ne, atol=1e-1)

import matplotlib.pyplot as plt

plt.plot(cost_history, label="Batch GD")
plt.plot(cost_history_sgd, label="SGD")
plt.xlabel("Iteration")
plt.ylabel("Cost J(θ)")
plt.legend()
plt.savefig("lectures/lec02/cost_history.png", dpi=150)
plt.show()
