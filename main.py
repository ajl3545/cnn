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

#i = Image.open("tests/squid.jpg").convert("L")
#image_in = np.array(i)

conv = c.Convolutional(
    input=image_in,
    l1=c.Filter(matrix=[
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]),
    l2=c.MaxPool(size=2),
    l3=c.ReLU()

    #    ...

    #ln = whatever

)
out = conv.start()

# Show result of convulotional abstraction
plt.title("Input")
plt.imshow(image_in, cmap='gray')
plt.show()

plt.title("Out")
plt.imshow(out, cmap='gray')
plt.show()
