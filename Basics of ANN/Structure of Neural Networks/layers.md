# 1.3 Structure of Neural Networks

Neural networks are a fundamental building block of deep learning models. The structure of a neural network typically consists of several key components. Below is an overview of the structure, including the roles of different layers, types of connections, and the role of weights and biases.

## Layers

### 1. **Input Layer**
- The **input layer** is where the data enters the neural network.
- Each node in this layer corresponds to a feature from the dataset.
- It doesn't perform any computation, but simply passes the data to the next layer.

### 2. **Hidden Layers**
- **Hidden layers** are the layers between the input and output layers. They perform computations on the data.
- Neural networks can have one or more hidden layers. The number of hidden layers and neurons in each layer is a key hyperparameter.
- **Activation functions** are applied in hidden layers to introduce non-linearity into the model, enabling it to learn complex patterns.

### 3. **Output Layer**
- The **output layer** is the final layer of the neural network and produces the result.
- The output layer depends on the type of problem:
  - For **regression** tasks, it might have a single node with a linear activation function.
  - For **classification** tasks, it may use an activation function like **softmax** (for multi-class) or **sigmoid** (for binary classification).

## Dense Layers vs. Sparse Connections

### 1. **Dense Layers**
- A **dense layer** (also known as a **fully connected layer**) is a layer where every neuron is connected to every neuron in the previous layer.
- Each connection has an associated **weight**, and each neuron has a **bias**.
- Dense layers are common in feedforward neural networks and deep neural networks because they allow for powerful learning from the data.

### 2. **Sparse Connections**
- In contrast to dense layers, **sparse connections** have fewer connections between neurons. For example:
  - **Convolutional layers** in Convolutional Neural Networks (CNNs) are sparse, as each neuron is only connected to a small region of the input.
  - **Recurrent connections** in Recurrent Neural Networks (RNNs) allow neurons to share information over time.
- Sparse connections reduce the computational complexity and help models generalize better, especially for specific tasks like image processing or sequence modeling.

## Role of Weights and Biases

### 1. **Weights**
- **Weights** represent the strength of the connection between two neurons.
- During training, the weights are updated based on the gradients of the loss function. This process is what allows the neural network to learn patterns in the data.
- Weights are initialized randomly at the start of training and are refined through gradient descent or other optimization algorithms.

### 2. **Biases**
- **Biases** are additional parameters added to the neurons to allow the model to better fit the data.
- A bias is added to the weighted sum of inputs before passing it through the activation function. It helps the model to learn offsets or shifts in the data, allowing for better flexibility and accuracy in the learning process.
- Biases are also initialized randomly and updated during training.

## Summary

- Neural networks consist of **input**, **hidden**, and **output layers**, with each layer performing specific functions.
- **Dense layers** have fully connected neurons, while **sparse connections** (e.g., CNNs and RNNs) are used to reduce complexity and improve model performance for specific tasks.
- **Weights** and **biases** are key parameters that are learned and adjusted during the training process to minimize the error in predictions.

Understanding the structure of neural networks is crucial for designing effective models that can solve complex tasks, from image recognition to language translation.
