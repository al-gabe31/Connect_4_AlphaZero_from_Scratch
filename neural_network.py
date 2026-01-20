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
            alias: str = '',
            preceding_conns = None,
            suceeding_conns = None,
            activation_function = lambda x: x,
            bias: float = None,
        ):
        self.value = value # value head of this node
        self.alias = alias # what this node will be called
        if preceding_conns is None:
            self.preceding_conns = []
        else:
            self.preceding_conns = preceding_conns # list of nodes that connect to this object and its corresponding weights
        if suceeding_conns is None:
            self.suceeding_conns = []
        else:
            self.suceeding_conns = suceeding_conns # list of nodes this object connects to and its corresponding weights
        self.activation_function = activation_function # activation funciton to achieve non-linearity
        self.bias = bias # bias term when calculating value head

    def __str__(self):
        return f'{self.alias}: {self.value}'
    
    def __repr__(self):
        return f'{self.alias}: {self.value}'

    def connect_preceding_nodes(self, node_list, weight_list):
        for i in range(len(node_list)):
            # adding new connections for preceding nodes
            self.preceding_conns.append([node_list[i], weight_list[i]])

            # making sure that connection is also updated for that preceding node
            node_list[i].suceeding_conns.append([self, weight_list[i]])

        # don't forget to update value with the new connections
        self.value = self.calc_value()

    def connect_suceeding_nodes(self, node_list, weight_list):
        for i in range(len(node_list)):
            # adding new connections for suceeding nodes
            self.suceeding_conns.append([node_list[i], weight_list[i]])

            # making sure that connection is also updated for that succeeding node
            node_list[i].preceding_conns.append([self, weight_list[i]])

            # don't forget to update value of the suceeding node
            node_list[i].value = node_list[i].calc_value()
        
    def calc_value(self):
        summation = 0

        # going through each precedding connection and add up: value * weight
        for i in range(len(self.preceding_conns)):
            summation += self.preceding_conns[i][0].value * self.preceding_conns[i][1]

        # feed the summation into the activation function
        return self.activation_function(summation)