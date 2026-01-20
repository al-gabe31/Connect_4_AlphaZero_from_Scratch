# contains all classes for a neural network from sctach

import math

def relu(value):
    if value <= 0:
        return 0
    else:
        return value
    
# redeclaring built-in tanh just so it'll work when passing it as an argument into the node
def tanh(value):
    return math.tanh(value)

class Node:
    def __init__(
            self, 
            value: float = None,
            preceding_conns = [],
            suceeding_conns = [],
            activation_function: function = lambda x: 0,
            bias: float = None,
            preceding_conns_weights = [],
            suceeding_conns_weights = [],
        ):
        self.value = value # value head of this node
        self.preceding_conns = preceding_conns # list of nodes that connect to this object
        self.suceeding_conns = suceeding_conns # list of nodes this object connects to
        self.activation_function = activation_function # activation funciton to achieve non-linearity
        self.bias = bias # bias term when calculating value head
        self.preceding_conns_weights = preceding_conns_weights # list of weights from the nodes that connect to this object
        self.suceeding_conns_weights = suceeding_conns_weights # list of weights this object connects to other nodes

    def calc_value(self):
        summation = 0

        # going through each precedding connection and add up: value * weight
        for i in range(len(self.preceding_conns)):
            summation += self.preceding_conns[i].value * self.preceding_conns_weights[i]

        # feed the summation into the activation function
        return self.activation_function(summation)
    
    def update_weights(self):
        pass # little update