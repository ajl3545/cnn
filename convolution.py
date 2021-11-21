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

        output = None

        # A forward pass through all layers
        for x in self.layers.keys():
            output = self.layers[x].forward((self.img))

        return output


class Filter:  # Used to Convolve (finds features)
    def __init__(self, size):
        self.size = size
        self.W = np.ones(size*size)
        self.W = self.W.reshape((self.size, self.size))

    def forward(self, img):
        print(self.W)


class MaxPool:
    def __init__(self, size):
        self.size = size

    # Default stride of 1
    def forward(self, img):
        dim = len(img)-self.size+1
        down_sample = np.ones(dim*dim)
        down_sample = down_sample.reshape((dim, dim))

        for row in range(len(img)-self.size):
            for col in range(len(img[0])-self.size):
                sample = img[row:row+self.size, col:col+self.size]
                max = np.matrix(sample).max()
                down_sample[row][col] = max

        return down_sample

    # MaxPool back propogation keeps track of the largest
    # element index from the forward pass and uses it
    # as the gradient

    def backward(self):
        pass


# Load data
mnist = MNIST('MNIST/')
x_train, y_train = mnist.load_training()

# Get first Input image and reshape to np.array of (28x28) pixels
image = x_train[0]
image = np.array(image, dtype='int')
image_in = image.reshape((28, 28))

# Establish network with input, output, and layers
conv = Convolutional(
    input=image_in,
    # Layers
    l1=MaxPool(size=2),
    l2=Filter([  # Sharpen
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ])
)
image_out = conv.start()

# Show result of convulotional abstraction
plt.imshow(image_in, cmap='gray')
plt.show()
plt.imshow(image_out, cmap='gray')
plt.show()
