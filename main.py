import numpy as np
from matplotlib import pyplot as plt
from mnist import MNIST
from numpy.lib.arraypad import pad
import convolution as c
from PIL import Image

# Load data
mnist = MNIST('MNIST/')
x_train, y_train = mnist.load_training()

# Get first Input image and reshape to np.array of (28x28) pixels
image = x_train[123]
image = np.array(image, dtype='int')
image_in = image.reshape((28, 28))

i = Image.open("tests/test.jpg").convert("L")
image_in = np.array(i)

# Establish network with input, output, and layers
conv = c.Convolutional(
    # Input Layer
    input=image_in,
    # Convolutional Layers
    l1=c.Filter(matrix=[
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]),
)
image_out = conv.start()

# Show result of convulotional abstraction
ax = plt.gca()
ax.set_xticks(np.arange(-.5, 100, 1))
ax.set_yticks(np.arange(-.5, 100, 1))
plt.grid()


plt.title("Input(28x28) " + str(y_train[123]))
plt.imshow(image_in, cmap='gray')
plt.show()

ax = plt.gca()
ax.set_xticks(np.arange(-.5, 100, 1))
ax.set_yticks(np.arange(-.5, 100, 1))
plt.grid()

plt.title("Output " + str(y_train[123]))
plt.imshow(image_out, cmap='gray')
plt.show()
