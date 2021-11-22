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
        down_sample = np.zeros((dim-self.size+1)*(dim-self.size+1))
        down_sample = down_sample.reshape((dim-self.size+1, dim-self.size+1))
        down_sample = np.pad(down_sample, pad_width=0)

        for row in range(dim-self.size+1):
            for col in range(dim-self.size+1):
                sample = img[row: row+self.size, col: col+self.size]
                filtrix = np.zeros(sample.shape)
                for row_1 in range(len(sample)):
                    for col_1 in range(len(sample)):
                        filtrix[row_1][col_1] = sample[row_1][col_1] * \
                            self.matrix[row_1][col_1]
                down_sample[row][col] = np.sum(filtrix)

        return down_sample

    def backward(self):
        pass


class MaxPool:
    def __init__(self, size):
        self.size = size

    def forward(self, img):

        # Cache input for backprop
        self.img = img

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

        # Reconstruct the input image
        gradient = np.zeros(self.img.shape)

        # Fill in regions where the max value was

        row_count = len(self.img) // self.size
        col_count = len(self.img[0]) // self.size

        stride = self.size

        for row in range(row_count):
            for col in range(col_count):

                # Sub sample reigon of image to down sample from
                sample = self.img[row*stride:row*stride +
                                  stride, col*stride:col*stride+stride]

                # index of max value relative to sample matrix (self.size x self.size)
                argmax = np.argmax(sample)

                # Find relative position in sample matrix
                y = argmax // len(sample)
                x = argmax % len(sample)

                # Value to pass to gradient
                max = sample[y][x]

                # get index pos in gradient image
                gradient[y+(row*stride)][x+(col*stride)] = max

        return gradient


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

    def backward(self):
        pass


class ReLU:

    def forward(self, img):

        self.image = img

        for row in range(len(img)):
            for col in range(len(img[0])):
                if img[row][col] < 0:
                    self.image[row][col] = 0
                else:
                    self.image[row][col] = img[row][col]

        return self.image

    def backward(self):

        derive = np.zeros(self.image.shape)

        for row in range(len(derive)):
            for col in range(len(derive[0])):
                if self.image[row][col] > 0:
                    derive[row][col] = 1

        return derive
