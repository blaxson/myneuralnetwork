from graphics import *
from network import NeuralNetwork, sigmoid
from data_loader import load_data_wrapper
import time 

""" used exclusively for the NN representation """
class Neuron(object):
    def __init__(self, x, y, bias):
        self.position = Point(x, y)
        self.circle = Circle(self.position, 17.5)
        self.bias = bias
        self.circle.setFill("white")
        self.circle.setOutline("black")

    def draw(self, win):
        self.circle.draw(win)
        # if self.bias:
        #     txt = Text(self.position, self.bias)
        #     txt.draw(win)

""" used exclusively for the NN representation """
class Layer(object):
    def __init__(self, index, size, weights, biases):
        self.index = index #index of Layer relative to NN
        self.num_neurons = size
        self.neurons = [None] * size
        self.weights = weights
        self.biases = biases

    """ given a layer object, update_layer updates the weights and biases of 
    that layer and returns the updated layer """
    def update_layer(self, weights, biases):
        self.weights = weights
        self.biases = biases

    """ given the x and y coordinates and the index of the neuron in the layer,
        set_neuron creates a neuron and adds it to the layer's neurons list. 
        Returns the new neuron object """
    def set_neuron(self, x_val, y_val, n_index):
        bias = None
        if self.biases is not None:
            bias = self.biases[n_index]
        n = Neuron(x_val, y_val, bias)
        self.neurons[n_index] = n
        return n 

    """ sets the neurons in the Layer object of layers with more than 16 neurons
    """
    def set_large_layer(self, win, x_val):
        for i in range (0, 8):
            self.set_neuron(x_val, 40 + (i*45), i)
        for i in range (0, 8):
            n_index = self.num_neurons - (1 + i)
            self.set_neuron(x_val, 760 - (i*45), n_index)

    """ sets the neurons in the Layer object of layers with 16 or less neurons
    """
    def set_small_layer(self, win, x_val):
        mid = self.num_neurons // 2
        if self.num_neurons % 2: # if odd, then add middle neuron first
            self.set_neuron(x_val, 400, mid)
            for i in range (0, mid):
                n_index = mid - (i + 1)
                self.set_neuron(x_val, 355 - (i*45), n_index)
            for i in range(0, mid):
                n_index = mid + (i + 1)
                self.set_neuron(x_val, 445 + (i*45), n_index)
        else:
            for i in range(0, mid):
                n_index = mid - (i + 1)
                self.set_neuron(x_val, 377.5 - (i*45), n_index)
            for i in range(0, mid):
                n_index = mid + i
                self.set_neuron(x_val, 422.5 + (i*45), n_index)

    def draw_large_layer(self, win):
        x_val = self.neurons[0].position.getX()
        pt =  Point(x_val, 400)
        pt.draw(win)
        pt = Point(x_val, 390)
        pt.draw(win)
        pt = Point(x_val,410)
        pt.draw(win)
        for i in range(0, 8):
            self.neurons[i].draw(win) # draw first 8 neurons
            self.neurons[self.num_neurons - (1 + i)].draw(win) # last 8 neurons

    def draw_small_layer(self, win):
        # can draw all neurons at once b/c pos already known & no dots in middle
        for i in range(0, self.num_neurons):
            self.neurons[i].draw(win) 
    
    """ for following graphics.py style and structure """
    def draw(self, win):
        if self.num_neurons > 16:
            self.draw_large_layer(win)
        else:
            self.draw_small_layer(win)
# ******************************************************************************

def main():
    training_data, validation_data, test_data = load_data_wrapper()
    network = NeuralNetwork([784, 16, 16, 10])
    
    layers = create_layers(network)
    win, wth = initialize_screen(network.sizes)
    draw_network(win, layers) # draw initial network with random weights
    
    epochs = 30
    batch_size = 10
    learning_rate = 3.0

    txt = Text(Point(wth/2, 830), "Initial Weights")
    txt.draw(win)
    # main training loop
    for i in range(epochs): # for each iteration of training
        biases, weights = network.train_iteration(training_data, batch_size, \
        learning_rate, i, test_data=test_data)
        txt.setText("Iteration: {0}".format(i))
        for j in range(1, len(layers)):
            layers[j].update_layer(weights[j-1], biases[j-1])
        draw_network(win, layers)

    win.getMouse()
    win.close()

""" takes in NeuralNetwork Object and returns a list of layers, where each 
    index corresponds to each proceeding layer. layer[0] is first layer in nn, 
    layer[1] is second, etc. """
def create_layers(nn):
    # creating a list of layers to use for nnDisplay
    layers = []
    layers.append(Layer(0, nn.sizes[0], None, None))
    for i in range(0, nn.num_layers - 1):
        #initial layer doesn't have weights or biases b/c no prev. layer
        layers.append(Layer(i+1, nn.sizes[i+1], nn.weights[i], nn.biases[i]))
    return layers


def initialize_screen(sizes):
    wth = 160 + (280 * (len(sizes)-1))
    wth = wth if wth < 1280 else 1280
    win = GraphWin(title="my window", width=wth, height=900)
    win.setBackground("white")
    ln = Line(Point(0,800), Point(wth, 800))
    ln.draw(win)
    return win, wth

def draw_network(win, layers):
    for i in range(0, len(layers)):
        set_layer(win, 80 + (i * 280), layers[i])
        if i > 0:
            draw_weights(win, layers[i], layers[i-1])
            layers[i-1].draw(win) # draw neurons after drawing weight lines
    layers[len(layers)-1].draw(win) # draw last layer of neurons

def set_layer(win, x_val, layer):
    if layer.num_neurons > 16:
        layer.set_large_layer(win, x_val)
    else:
        layer.set_small_layer(win, x_val)

def update_line(win, cur_layer, prev_layer, i, j):
    line = Line(cur_layer.neurons[i].position, prev_layer.neurons[j].position)
    if cur_layer.weights[i][j] < 0:
        line.setOutline("red")
    else:
        line.setOutline("blue")
    line.setWidth(8 * (sigmoid(abs(cur_layer.weights[i][j])) - 0.5))
    line.draw(win)

def draw_weights(win, cur_layer, prev_layer):
    cur_mid = cur_layer.num_neurons // 2
    prev_mid = prev_layer.num_neurons // 2
    # set the first neurons after the mid... accounts for large layers
    if cur_layer.num_neurons > 16:
        print(cur_layer.num_neurons)
        end_cur = cur_layer.num_neurons - 9
    else:
        end_cur = cur_mid - 1
    if prev_layer.num_neurons > 16:
        end_prev = prev_layer.num_neurons - 9
    else:
        end_prev = prev_mid - 1
    
    print("end_cur: {0}".format(end_cur))
    print("end_prev: {0}".format(end_prev))
    
    # do the 1st half of current layer
    for i in range(0, min(cur_mid, 8)):
        # connects w/ 1st half of prev_layer
        for j in range(0, min(prev_mid, 8)): 
            update_line(win, cur_layer, prev_layer, i, j)
        # connects w/ 2nd half of prev_layer
        for j in range(prev_layer.num_neurons - 1, end_prev, -1):
            update_line(win, cur_layer, prev_layer, i, j)
    # do the 2nd half of current layer
    for i in range(cur_layer.num_neurons - 1, end_cur, -1):
        # connects w/ 1st half of prev_layer
        for j in range(0, min(prev_mid, 8)): 
            update_line(win, cur_layer, prev_layer, i, j)
        # connects w/ 2nd half of prev_layer
        for j in range(prev_layer.num_neurons - 1, end_prev, -1):
            update_line(win, cur_layer, prev_layer, i, j)


main()