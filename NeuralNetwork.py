import random
import numpy as np

class NeuralNetwork(object):
   
    """ The list 'sizes' contains the num of neurons in the respective layers of 
        the neural network. If the list is [24, 16, 10], then it would be a 3-layer
        NN with 24 nodes in the first layer, 16 in the second, etc. """
    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        # biases and weights stored as lists of matrices
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # example: net.weights[0] is a matrix storing the weights connecting
        # the 1st and 2nd layers of neurons.. net.weights[1] is 2nd & 3rd, etc.
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        # with W = net.weights[1], W_jk connects Kth neuron in second layer 
        # to Jth neuron in third layer
        
    """ returns the output of the neural network given "x" input """
    def feedforward(self, x):
        # go thru each matrix of weights & biases in list (eq. to each layer)
        for bias, weight in zip(self.biases, self.weights):
            # based off of feed-forward equation x' = sig(weight*x + bias)
            x = sigmoid(np.dot(weight, x) + bias)
        return x 

    """ Trains the neural network using stochastic gradient descent using
        dataset batches. 'trainging_data' is a list of tuples (x, y) containing 
        the 'x' training inputs and the 'y' correct desired outputs. 'epochs' is 
        the number of desired iterations/cycles to train for. 'batch_size' is
        the desired size of the randomly chosen dataset batches. 'l_rate' is the
        learning rate of the neural network. Optional arg 'test_data' is given 
        which will evaluate the network after each epoch and print partial
        progress (warning: very slow) """
    def SGD(self, training_data, epochs, batch_size, l_rate, test_data=None):

        if test_data:
            len_test = len(test_data)
        len_training = len(training_data)
        # for each iteration of training
        for i in range(epochs):
            # randomly shuffle data to help with training process
            random.shuffle(training_data)
            # partition all data into batches of data
            batches = [
                training_data[j : j + batch_size] for j in range(0, len_training, batch_size)]
            # go thru each batch & apply gradient decent in backpropogation
            for batch in batches:
                self.update_weights_and_biases(batch, l_rate)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    i, self.evaluate(test_data), len_test))
            else:
                print("Epoch {0} complete".format(i)) 
        
    """ updates the neural network's weights and biases by applying gradient
        descent using backpropogation on a single batch of test data. 'batch'
        is a list of tuples (x, y) and 'l_rate' is the learning rate """
    def update_weights_and_biases(self, batch, l_rate):
        gradient_b = [np.zeros(bias.shape) for bias in self.biases]
        gradient_w = [np.zeros(weight.shape) for weight in self.weights]
        for x, y in batch:
            delta_gradient_b, delta_gradient_w = self.backpropogate(x, y)
            gradient_b = [gb+dgb for gb, dgb in zip(gradient_b, delta_gradient_b)]
            gradient_w = [gw+dgw for gw, dgw in zip(gradient_w, delta_gradient_w)]
        self.weights = [w - (l_rate / len(batch)) * gw 
                        for w, gw in zip(self.weights, gradient_w)]
        self.biases =  [b - (l_rate / len(batch)) * gb 
                        for b, gb in zip(self.biases, gradient_b)]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))