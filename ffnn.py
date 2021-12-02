import numpy as np
from numpy import random

# STEPS
# Layers Make up each step
# At each layer is a set of Nodes
# Each node sums each of the weights multiplied by the values (features input from instantiation)
# Each node then activates the summation and passes it to the next node layer
# At each hidden layer, a ReLU will activate
# Once done time to classify
#   Multi-class with softmax
#   might be worth doing a multi-label classification for the learning and voting process
# calcuate error
# propogate error through gradients


class FFNN:
    # layers = number of hidden layers
    # self.layers = [x1,x2,x3]; x1 = length of feature map
    # each index corresponds to the number of nodes
    def __init__(self, features, layers):
        self.features = features
        self.layers = layers
        self.network = self.construct_network()

    def construct_network(self):
        network = []
        layer = []
        # First hidden layer
        for i in range(self.layers[0]):
            n = Node(self.random_weights(len(self.features)))
            n.inputs = self.features
            layer.append(n)
        network.append(layer)

        for i in range(len(self.layers)-1):  # All but first
            # next layer, since we dont wan't first
            amt_nodes = self.layers[i+1]
            layer = []
            for j in range(amt_nodes):
                n = Node(self.random_weights(self.layers[i]))  # previous layer
                layer.append(n)
            network.append(layer)

        return network

    def random_weights(self, amt):
        return np.random.randint(0, 9, amt)

    def backward(self):
        # get loss on pre diction,
        # derive softmax
        # propogate backwards thru FFNN network
        pass

    # Softmax: is used before classification
    def activate(self):
        # implement
        pass


class Node:
    def __init__(self, weights):
        self.weights = weights
        self.inputs = []
        self.bias = 1

    def forward(self, inputs):
        self.inputs = inputs
        sum = 0
        for x in range(len(self.inputs)):
            sum += self.inputs[x]*self.weights[x]
        return self.activate(sum+self.bias)

    # Since this node takes in many weights/inputs,
    # the gradient will be a vector representing the loss of each term
    def backward(self, lr, dL):

        # The partial derivative of the linear combination for each term
        # for i=0 to len(weights) -> (dL/d_wi) = xi+1; where 1 is the bias term
        gradient = np.copy(self.inputs)

        # Then multiplied by the relu partial derivative
        # local_gradient = (d_wi/dL) * d_relu
        for i in range(len(gradient)):
            local_gradient = gradient[i] * (1 * (gradient[i] > 0)) + 1
            upstream_gradient = dL
            gradient[i] = (local_gradient * upstream_gradient)
            self.weights[i] -= lr*gradient[i]

        return gradient

    def activate(self, value):
        if value < 0:
            return 0
        return value


network = FFNN(
    features=np.ones(25),  # Input features
    layers=[16, 16, 10],
)

# Test to see if the network constructs itself
print("Input layer: " + str(len(network.features)))
count = 0
for x in network.features:
    print("Feature" + str(count) + ": " + str(x))
    count += 1
layer_count = 0
for layer in network.network:
    print("Layer " + str(layer_count) + " len=" + str(len(layer)))
    layer_count += 1
    for node in layer:
        print(str(node) + str(len(node.weights)))
