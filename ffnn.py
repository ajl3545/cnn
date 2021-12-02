import numpy as np

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
    def __init__(self, inputs, layers, hidden_activation, output_activation):
        self.i = inputs
        self.layers = layers
        # function for calculating the hidden activations
        self.hidden_activation = hidden_activation
        # function for calculating the output activation
        self.output_activation = output_activation

    def construct_network(self):
        pass

    # hidden activation
    def ReLU(self):
        pass

    # output activation
    def softmax(self):
        pass


class Node:
    def __init__(self, inputs):
        self.inputs = inputs
        self.weights = np.ones(len(self.inputs))
        self.bias = 1
