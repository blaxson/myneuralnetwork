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
    def train(self, training_data, epochs, batch_size, l_rate, test_data=None):

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
                print("Iteration {0}: {1} / {2}".format(
                    i, self.evaluate(test_data), len_test))
            else:
                print("Iteration {0} complete".format(i)) 
        
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
    
    def backpropogate(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_deriv(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_deriv(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))