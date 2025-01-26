# File: Weighted_sum_and_bias.py

# --- Theoretical Explanation ---
# In the context of neural networks, a "weighted sum" is a fundamental computation. It combines inputs
# with their associated weights to produce a single scalar output. Mathematically, it's represented as:
#   weighted_sum = w1 * x1 + w2 * x2 + ... + wn * xn
# where:
# - w1, w2, ..., wn are weights (parameters learned during training),
# - x1, x2, ..., xn are the inputs.
# A bias is an additional parameter added to the weighted sum. It shifts the output, allowing the model
# to represent data that does not pass through the origin. The equation becomes:
#   output = weighted_sum + bias = (w1 * x1 + w2 * x2 + ... + wn * xn) + b

# --- Practical Example ---
# Let's calculate the weighted sum and bias for a single-layer neural network with two inputs.

# Define inputs
inputs = [1.5, -2.0]  # x1 and x2

# Define weights
weights = [0.8, 1.2]  # w1 and w2

# Define bias
bias = 0.5  # b

# Calculate weighted sum
weighted_sum = sum(w * x for w, x in zip(weights, inputs))

# Add bias to the weighted sum
output = weighted_sum + bias

# Print the results
print("Weighted Sum:", weighted_sum)
print("Output (Weighted Sum + Bias):", output)

# --- Theoretical Explanation ---
# In machine learning, weights and biases are adjusted during training using optimization algorithms
# (like gradient descent) to minimize the error between predicted and actual outputs. Weights control
# the impact of each input on the final output, while biases allow the model to shift the activation
# function, enabling it to fit the data better.

# A single-layer neural network with one neuron can only learn linear relationships. However, in real-world
# problems, data is often non-linear. To model such relationships, multiple layers of neurons (deep networks)
# and non-linear activation functions are used.

# --- Practical Example ---
# Let's extend the concept to a small neural network with three neurons in the first layer. Each neuron
# produces a weighted sum and bias calculation.

# Define inputs
inputs = [1.0, 2.0, 3.0]  # x1, x2, x3

# Define weights for three neurons
weights = [
    [0.2, 0.8, -0.5],  # Weights for neuron 1
    [0.5, -0.91, 0.26],  # Weights for neuron 2
    [-0.26, -0.27, 0.17],  # Weights for neuron 3
]

# Define biases for three neurons
biases = [2.0, 3.0, 0.5]  # Biases for neuron 1, 2, and 3

# Calculate outputs for each neuron
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    # Weighted sum for the current neuron
    neuron_output = sum(w * x for w, x in zip(neuron_weights, inputs)) + neuron_bias
    layer_outputs.append(neuron_output)

# Print the outputs of all neurons in the layer
print("Layer Outputs:", layer_outputs)

# --- Theoretical Explanation ---
# The outputs calculated above can be passed to an activation function to introduce non-linearity.
# Common activation functions include:
# - ReLU (Rectified Linear Unit): max(0, x)
# - Sigmoid: 1 / (1 + exp(-x))
# - Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
# Activation functions enable neural networks to learn complex non-linear mappings.

# --- Practical Example ---
# Let's apply the ReLU activation function to the outputs of the layer.


def relu(x):
    return max(0, x)


# Apply ReLU to each output
activated_outputs = [relu(output) for output in layer_outputs]

# Print the activated outputs
print("Activated Outputs (ReLU):", activated_outputs)
