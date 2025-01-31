# Linear vs. Non-linear Models
# 1. Linear Models: These models assume a linear relationship between input variables (features)
# and the single output variable.
#    Equation: y = w1*x1 + w2*x2 + ... + wn*xn + b
#    Example: Linear Regression, Logistic Regression (with a linear decision boundary).

# 2. Non-linear Models: These models capture non-linear relationships between input variables
# and the output variable.
#    Equation: y = f(x1, x2, ..., xn) where f is a non-linear function.
#    Example: Decision Trees, Neural Networks, Polynomial Regression.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate synthetic data
np.random.seed(0)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 2

# Linear Model
# Create a linear regression model
linear_model = LinearRegression()
# Step 1: Train the model
linear_model.fit(X, y)
# Step 2: Predict using the same X values
y_pred_linear = linear_model.predict(X)

# Non-linear Model (Polynomial Regression)
poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
poly_model.fit(X, y)
y_pred_poly = poly_model.predict(X)

# Plotting the results
plt.scatter(X, y, color="black", label="Data")
plt.plot(X, y_pred_linear, color="blue", label="Linear Model")
plt.plot(X, y_pred_poly, color="red", label="Non-linear Model (Polynomial)")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear vs. Non-linear Models")
plt.legend()
plt.show()
