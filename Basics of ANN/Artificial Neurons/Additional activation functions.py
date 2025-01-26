# 1. Leaky ReLU: A variant of ReLU that allows a small, non-zero gradient for negative inputs.
#    Equation: f(x) = x if x > 0 else alpha * x (where alpha is a small constant, e.g., 0.01).
# 2. Swish: A smooth, non-monotonic function that tends to work better for deep networks.
#    Equation: f(x) = x / (1 + exp(-x)).
# 3. GELU (Gaussian Error Linear Unit): A smoother approximation of the ReLU function.
#    Equation: f(x) = x * 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).

# --- Practical Explanation with Matplotlib ---
# Let's visualize these activation functions to understand their behavior.

import numpy as np
import matplotlib.pyplot as plt


# Define Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


# Define Swish
def swish(x):
    return x / (1 + np.exp(-x))


# Define GELU
# Approximation as defined in the theoretical equation
def gelu(x):
    return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# Generate input values for visualization
x = np.linspace(-5, 5, 500)

# Compute activation values
leaky_relu_values = leaky_relu(x)
swish_values = swish(x)
gelu_values = gelu(x)

# Plot the activation functions
plt.figure(figsize=(10, 6))

# Plot Leaky ReLU
plt.plot(x, leaky_relu_values, label="Leaky ReLU", color="blue")

# Plot Swish
plt.plot(x, swish_values, label="Swish", color="green")

# Plot GELU
plt.plot(x, gelu_values, label="GELU", color="red")

# Add labels, legend, and grid
plt.title("Activation Functions: Leaky ReLU, Swish, and GELU")
plt.xlabel("Input (x)")
plt.ylabel("Output (f(x))")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.grid(alpha=0.3)
plt.legend()

# Show the plot
plt.show()

# --- Theoretical Explanation ---
# Observations from the plots:
# 1. Leaky ReLU: Allows small negative outputs, mitigating the dying neuron problem of ReLU.
# 2. Swish: Smooth and non-monotonic, which helps improve gradient flow in deep networks.
# 3. GELU: Smooth and differentiable, making it suitable for modern architectures like transformers.

# These functions are often selected based on the specific use case and network architecture.
