import numpy as np
from matplotlib import pyplot as plt
from mnist import MNIST
import convolution as c

# Load data
mnist = MNIST('MNIST/')
x_train, y_train = mnist.load_training()

# Get first Input image and reshape to np.array of (28x28) pixels
image = x_train[123]
image = np.array(image, dtype='int')
image_in = image.reshape((28, 28))

# Establish network with input, output, and layers
conv = c.Convolutional(
    # Input Layer
    input=image_in,
    # Convolutional Layers
    l1=c.MaxPool(size=2),
)

image_out = conv.start()

# Show result of convulotional abstraction
plt.title("Input(28x28) " + str(y_train[123]))
plt.imshow(image_in, cmap='gray')
plt.show()
plt.title("Output " + str(y_train[123]))
plt.imshow(image_out, cmap='gray')
plt.show()
