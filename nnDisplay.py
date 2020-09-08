from graphics import *
from network import NeuralNetwork
from data_loader import load_data_wrapper

""" used exclusively for the NN representation """
class Neuron(object):
    def __init__(self, x, y, bias):
        self.position = Point(x, y)
        self.circle = Circle(self.position, 17.5)
        self.bias = bias

    def draw_neuron(self, win):
        self.circle.draw(win)
        if self.bias:
            txt = Text(self.position, self.bias)
            txt.draw(win)

""" used exclusively for the NN representation """
class Layer(object):
    def __init__(self, index, size, weights, biases):
        self.index = index #index of Layer relative to NN
        self.num_neurons = size
        self.neurons = [None] * size
        self.weights = weights
        self.biases = biases

    """ given the x and y coordinates and the index of the neuron in the layer,
        set_neuron creates a neuron and adds it to the layer's neurons list. 
        Returns the new neuron object """
    def set_neuron(self, x_val, y_val, index):
        bias = None
        if self.biases is not None:
            bias = self.biases[index]
        n = Neuron(x_val, y_val, bias)
        self.neurons[index] = n
        return n 

    def set_large_layer(self, win, x_val):
        pt = Point(x_val, 400)
        pt.draw(win)
        pt = Point(x_val, 390)
        pt.draw(win)
        pt = Point(x_val,410)
        pt.draw(win)
        for i in range (0, 8):
            bias = None
            if self.biases is not None:
                bias = self.biases[i]
            
            n = Neuron(x_val, 40 + (i*45), bias)
            self.neurons[i] = n
            n.draw_neuron(win)

        for i in range (0, 8):
            n_index = self.num_neurons - (1 + i)
            bias = None 
            if self.biases is not None:
                bias = self.biases[n_index]

            n = Neuron(x_val, 760 - (i*45), bias)
            self.neurons[n_index] = n
            n.draw_neuron(win)

    def set_layer(self, win, x_val):
        mid = self.num_neurons // 2
        if self.num_neurons % 2: # if odd, then add middle neuron first
            bias = None 
            if self.biases is not None:
                bias = self.biases[mid]

            n = Neuron(x_val, 400, bias)
            self.neurons[mid] = n
            n.draw_neuron(win)

            for i in range (0, mid):
                n_index = mid - (i + 1)
                bias = None 
                if self.biases is not None:
                    bias = self.biases[n_index]

                n = Neuron(x_val, 355 - (i*45), bias)
                self.neurons[n_index] = n
                n.draw_neuron(win)
  
            for i in range(0, mid):
                n_index = mid + (i + 1)
                bias = None 
                if self.biases is not None:
                    bias = self.biases[n_index]

                n = Neuron(x_val, 445 + (i*45), bias)
                self.neurons[n_index] = n
                n.draw_neuron(win)

        else:
            for i in range(0, mid):
                n_index = mid - (i + 1)
                bias = None 
                if self.biases is not None:
                    bias = self.biases[n_index]

                n = Neuron(x_val, 377.5 - (i*45), bias)
                self.neurons[n_index] = n
                n.draw_neuron(win) 

            for i in range(0, mid):
                n_index = mid + i
                bias = None 
                if self.biases is not None:
                    bias = self.biases[n_index]

                n = Neuron(x_val, 422.5 + (i*45), bias)
                self.neurons[n_index] = n
                n.draw_neuron(win)

# ******************************************************************************

def main():
    training_data, validation_data, test_data = load_data_wrapper()
    network = NeuralNetwork([784, 16, 16, 10])
    layers = create_layers(network)
    win = initialize_screen(network.sizes)
    set_network(win, layers)
    win.getMouse()
    win.close()

def initialize_screen(sizes):
    wth = 160 + (280 * (len(sizes)-1))
    wth = wth if wth < 1280 else 1280
    win = GraphWin(title="my window", width=wth, height=900)
    win.setBackground("white")
    ln = Line(Point(0,800), Point(wth, 800))
    ln.draw(win)
    return win

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

def draw_layer(win, x_val, layer):
    if layer.num_neurons > 16:
        layer.set_large_layer(win, x_val)
    else:
        layer.set_layer(win, x_val)
    return None

def set_network(win, layers):
    for i in range(0, len(layers)):
        draw_layer(win, 80 + (i * 280), layers[i])
    return None

main()