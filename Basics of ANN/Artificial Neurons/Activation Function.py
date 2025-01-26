import numpy as np
import matplotlib.pyplot as plt

# Activation functions are critical in neural networks as they introduce non-linearity, enabling the
# network to learn complex relationships. Here, we'll discuss three additional activation functions:
""" 
Sigmoid function : 1/(1+e^(-x))
- definition
- range : (0,1)
- not zero centered
- causes vanishing gradient problem
- works well for binary classification

Tanh function : (e^x - e^(-x))/(e^x + e^(-x))
    - defination 
    - range : (-1,1)
    - zero centered
    - helps in centering the data by making the mean of the data close to zero
    - helps in learning the weights
    - helps in reducing the vanishing gradient problem

ReLU function : max(0,x)
    - definition
    - range : [0,∞)
    - computationally efficient
    - helps solve vanishing gradient problem
    - most commonly used in deep neural networks
    - can cause "dying ReLU" problem when neurons become permanently inactive
    
Softmax function : e^xi/∑e^xi
    - definition
    - range : (0,1)
    - used in multi-class classification problems
    - converts raw scores into probabilities
    - sum of all probabilities is 1
    - helps in making the model more confident about its predictions

"""


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Generate input values
input_values = np.linspace(-10, 10, 100)

# Apply sigmoid function
y = sigmoid(input_values)

# Plot sigmoid function
plt.plot(input_values, y)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()


# Tanh function


def tanh(x):
    return np.tanh(x)


input_values = np.linspace(-10, 10, 100)
y = tanh(input_values)

plt.plot(input_values, y)
plt.title("Tanh Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()


# ReLU function : max(0,x)
def relu(x):
    return np.maximum(0, x)


input_values = np.linspace(-10, 10, 100)
y = relu(input_values)

plt.plot(input_values, y)
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()


# Softmax function
def softmax(x):
    exp_values = np.exp(x)
    return exp_values / np.sum(exp_values)


input_values = np.array([1, 2, 3, 4, 5])
y = softmax(input_values)

# Plot softmax function
plt.bar(range(len(input_values)), y)
plt.title("Softmax Activation Function")
plt.xlabel("Input Index")
plt.ylabel("Probability")
plt.grid(True)
plt.show()
