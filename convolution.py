import matplotlib as plt
from mnist import MNIST


mnist = MNIST('MNIST/')
x_train, y_train = mnist.load_training()
x_test, y_test = mnist.load_testing()
