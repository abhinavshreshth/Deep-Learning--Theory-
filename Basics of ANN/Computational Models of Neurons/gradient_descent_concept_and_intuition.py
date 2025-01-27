# Gradient Descent: An optimization algorithm used to minimize the cost function in machine learning models.
# The goal is to find the parameters (weights) that minimize the cost function.
# The algorithm iteratively adjusts the parameters in the direction of the negative gradient of the cost function.

# 1. Initialize Parameters: Start with random values for the parameters (weights).
# 2. Compute Gradient: Calculate the gradient of the cost function with respect to each parameter.
# 3. Update Parameters: Adjust the parameters by moving in the direction opposite to the gradient.
#    Equation: theta = theta - alpha * gradient (where alpha is the learning rate).
# 4. Repeat: Iterate steps 2 and 3 until convergence (i.e., the cost function reaches a minimum).

import numpy as np


# Define the cost function (Mean Squared Error for simplicity)
def cost_function(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost


# Define the gradient descent function
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta = theta - learning_rate * gradient
        cost_history[i] = cost_function(X, y, theta)

    return theta, cost_history


# Example usage
# Generate some example data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a bias term (column of ones) to the input data
X_with_bias = np.c_[np.ones((100, 1)), X]

# Initialize parameters (weights) randomly
theta_initial = np.random.randn(2, 1)

# Set learning rate and number of iterations
learning_rate = 0.1
iterations = 1000

# Perform gradient descent
theta_optimal, cost_history = gradient_descent(
    X_with_bias, y, theta_initial, learning_rate, iterations
)

# Print the optimal parameters
print("Optimal parameters (weights) after performing gradient descent:", theta_optimal)

import matplotlib.pyplot as plt

# Plot the cost function history
plt.plot(range(iterations), cost_history, color="blue")
plt.title("Cost Function Convergence")
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.grid()
plt.show()
