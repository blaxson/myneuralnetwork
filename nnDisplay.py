from graphics import *
from network import NeuralNetwork
from data_loader import load_data_wrapper

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
    def set_neuron(self, x_val, y_val, n_index):
        bias = None
        if self.biases is not None:
            bias = self.biases[n_index]
        n = Neuron(x_val, y_val, bias)
        self.neurons[n_index] = n
        return n 

    def set_large_layer(self, win, x_val):
        pt = Point(x_val, 400)
        pt.draw(win)
        pt = Point(x_val, 390)
        pt.draw(win)
        pt = Point(x_val,410)
        pt.draw(win)
        for i in range (0, 8):
            self.set_neuron(x_val, 40 + (i*45), i)
        for i in range (0, 8):
            n_index = self.num_neurons - (1 + i)
            self.set_neuron(x_val, 760 - (i*45), n_index)

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
        
    def draw(self, win):
        if self.num_neurons > 16:
            self.draw_large_layer(win)
        else:
            self.draw_small_layer(win)
# ******************************************************************************

def main():
    training_data, validation_data, test_data = load_data_wrapper()
    network = NeuralNetwork([784, 16, 5, 16, 10])
    layers = create_layers(network)
    win = initialize_screen(network.sizes)
    set_network(win, layers)
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
    return win

def set_network(win, layers):
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


def draw_weights(win, cur_layer, prev_layer):
    cur_mid = cur_layer.num_neurons // 2
    prev_mid = prev_layer.num_neurons // 2
    # set the first neurons after the mid... accounts for large layers
    if cur_layer.num_neurons > 16:
        end_cur = cur_layer.num_neurons - 9
    else:
        end_cur = cur_mid - 1
    if prev_layer.num_neurons > 16:
        end_prev = prev_layer.num_neurons - 9
    else:
        end_prev = prev_mid - 1
    
    # do the first half of current layer
    for i in range(0, min(cur_mid, 8)):
        # connects with first half of prev_layer
        for j in range(0, min(prev_mid, 8)): 
            line = Line(cur_layer.neurons[i].position, prev_layer.neurons[j].position)
            line.draw(win)
        # connects with second half of prev_layer
        for j in range(prev_layer.num_neurons - 1, end_prev, -1):
            line = Line(cur_layer.neurons[i].position, prev_layer.neurons[j].position)
            line.draw(win)
    # do the second half of current layer
    for i in range(cur_layer.num_neurons - 1, end_cur - 1, -1):
        # connects with first half of prev_layer
        for j in range(0, min(prev_mid, 8)): 
            line = Line(cur_layer.neurons[i].position, prev_layer.neurons[j].position)
            line.draw(win)
        # connects with second half of prev_layer
        for j in range(prev_layer.num_neurons - 1, end_prev, -1):
            line = Line(cur_layer.neurons[i].position, prev_layer.neurons[j].position)
            line.draw(win)

main()