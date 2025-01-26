import numpy as np
import matplotlib.pyplot as plt

# Biological Neuron
"""

- The biological neuron is the basic building block of the human brain.
- The biological neuron consists of a cell body, dendrites, and an axon.
- The cell body contains the nucleus and other organelles.
- The dendrites receive signals from other neurons.
- The axon transmits signals to other neurons.
- The biological neuron communicates with other neurons through synapses.
- The synapse is the connection between two neurons.
- The synapse allows the transmission of signals from one neuron to another.
- The biological neuron receives signals from other neurons through synapses.
- The biological neuron integrates the signals it receives and generates an output signal.
- The output signal is transmitted to other neurons through the axon.
- The biological neuron can be in an active state or an inactive state.
- The biological neuron can change its state based on the signals it receives.
- The biological neuron can learn from experience and adapt to new situations.
- The biological neuron is the basic unit of information processing in the human brain.
- The biological neuron can perform complex computations and solve difficult problems.

"""

# Artificial Neuron
"""
- The artificial neuron is the basic building block of artificial neural networks.
- The artificial neuron is inspired by the biological neuron.
- The artificial neuron consists of an input layer, a processing unit, and an output layer.
- The input layer receives input signals from the environment.
- The processing unit integrates the input signals and generates an output signal.
- The output layer transmits the output signal to other neurons.
- The artificial neuron communicates with other neurons through connections.
- The connection is the link between two neurons.
- The connection allows the transmission of signals from one neuron to another.
- The artificial neuron receives input signals from the environment through connections.
- The artificial neuron integrates the input signals and generates an output signal.
- The output signal is transmitted to other neurons through connections.
- The artificial neuron can be in an active state or an inactive state.
- The artificial neuron can change its state based on the input signals it receives.
- The artificial neuron can learn from experience and adapt to new situations.
- The artificial neuron is the basic unit of information processing in artificial neural networks.
- The artificial neuron can perform complex computations and solve difficult problems.
- The artificial neuron can be trained to perform specific tasks and achieve high levels of performance.
- The artificial neuron is a powerful tool for building intelligent systems and developing advanced technologies.
- The artificial neuron is a key component of artificial intelligence and machine learning.
- The artificial neuron is the foundation of deep learning and neural network research.
- The artificial neuron is a versatile and flexible building block that can be used in a wide range of applications.
- The artificial neuron is a fundamental concept in the field of artificial intelligence and computer science.
- The artificial neuron is a revolutionary innovation that has transformed the way we think about intelligence and computation.
- The artificial neuron is a groundbreaking invention that has opened up new possibilities for creating intelligent machines and systems.
- The artificial neuron is a remarkable achievement that has revolutionized the field of artificial intelligence and machine learning.
- The artificial neuron is a remarkable breakthrough that has paved the way for the development of advanced technologies and intelligent systems.
- The artificial neuron is a remarkable discovery that has reshaped our understanding of intelligence and computation.
- The artificial neuron is a remarkable invention that has redefined the boundaries of what is possible in the field of artificial intelligence and machine learning.
- The artificial neuron is a remarkable milestone that has set the stage for the next generation of intelligent machines and systems.
- The artificial neuron is a remarkable innovation that has sparked a revolution in the field of artificial intelligence and machine learning.
- The artificial neuron is a remarkable advancement that has propelled us into a new era of intelligent systems and technologies.
- The artificial neuron is a remarkable achievement that has inspired a new wave of research and development in the field of artificial intelligence and machine learning.

"""


# --- Simple practical AI Neuron implementation to understand ---
class AINeuron:
    """
    A simple artificial neuron with a sigmoid activation function.

    Attributes:
        weights (numpy.ndarray): The weights of the neuron.
        bias (float): The bias of the neuron.
    """

    def __init__(self, n_inputs):
        # Initialize weights and bias randomly
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        # Compute weighted sum and apply activation function
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(weighted_sum)


# Create and test the neuron
neuron = AINeuron(3)
test_input = np.array([0.5, 0.8, 0.1])
output = neuron.forward(test_input)
print(f"AI Neuron Output: {output}")
