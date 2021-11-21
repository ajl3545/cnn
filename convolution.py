from matplotlib.colors import same_color
import numpy as np
from matplotlib import pyplot as plt
from mnist import MNIST


class Convolutional:  # Steps through each layer, extracts features, adjusts kernals
    def __init__(self, input, **layers):
        self.img = input
        self.layers = layers

    # Uses kernals and pooling layers to abstract features
    def start(self):

        output = self.img

        # A forward pass through all layers
        for x in self.layers.keys():
            output = self.layers[x].forward((output))

        return output


class Filter:  # Used to Convolve (finds features)
    def __init__(self, matrix):
        self.matrix = matrix
        self.size = len(matrix)

    def forward(self, img):
        dim = len(img)
        down_sample = np.ones(dim*dim)
        down_sample = down_sample.reshape((dim, dim))

        for row in range(len(img)-self.size+1):
            for col in range(len(img[0])-self.size+1):
                sample = img[row: row+self.size, col: col+self.size]
                filtrix = np.matmul(sample, self.matrix)
                down_sample[row: row+self.size, col: col+self.size] = filtrix

        return down_sample


class MaxPool:
    def __init__(self, size):
        self.size = size

    def forward(self, img):
        # max pool strides by its matrix size
        # this handles if the image is not neatly fitted
        # to the maxpool filter size

        row_count = len(img) // self.size
        col_count = len(img[0]) // self.size

        down_sample = np.ones(row_count*col_count)
        down_sample = down_sample.reshape((row_count, col_count))

        stride = self.size

        for row in range(row_count):
            for col in range(col_count):
                sample = img[row*stride:row*stride +
                             stride, col*stride:col*stride+stride]
                max = np.max(sample)
                down_sample[row][col] = max

        return down_sample

    # MaxPool back propogation keeps track of the largest
    # element index from the forward pass and uses it
    # as the gradient

    def backward(self):
        pass


class AveragePool:
    def __init__(self, size):
        self.size = size

    def forward(self, img):
        # max pool strides by its matrix size
        # this handles if the image is not neatly fitted
        # to the maxpool filter size

        row_count = len(img) // self.size
        col_count = len(img[0]) // self.size

        down_sample = np.ones(row_count*col_count)
        down_sample = down_sample.reshape((row_count, col_count))

        stride = self.size

        for row in range(row_count):
            for col in range(col_count):
                sample = img[row*stride:row*stride +
                             stride, col*stride:col*stride+stride]
                avg = np.sum(sample)/sample.size
                down_sample[row][col] = avg

        return down_sample

    # MaxPool back propogation keeps track of the largest
    # element index from the forward pass and uses it
    # as the gradient

    def backward(self):
        pass
