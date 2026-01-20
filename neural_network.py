# contains all classes for a neural network from sctach

import math

DEFAULT_WEIGHT = 1
DEFAULT_BIAS = 0

def relu(value):
    if value <= 0:
        return 0
    else:
        return value
    
# redeclaring built-in tanh just so it'll work when passing it as an argument into the node
def tanh(value):
    return math.tanh(value)

def sigmoid(value):
    return 1 / (1 + pow(math.e, -1 * value))

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
        self.preceding_conns = [] if preceding_conns is None else preceding_conns # list of nodes that connect to this object and its corresponding weights
        self.suceeding_conns = [] if suceeding_conns is None else suceeding_conns # list of nodes this object connects to and its corresponding weights
        self.activation_function = activation_function # activation funciton to achieve non-linearity
        self.bias = DEFAULT_BIAS if bias is None else bias # bias term when calculating value head

    def __str__(self):
        return f'{self.alias}: {self.value}'
    
    def __repr__(self):
        return f'{self.alias}: {self.value}'

    def connect_preceding_nodes(self, node_list, weight_list: list[float] = None, auto_update_values = False):
        for i in range(len(node_list)):
            # adding new connections for preceding nodes
            self.preceding_conns.append([node_list[i], DEFAULT_WEIGHT if weight_list is None else weight_list[i]])

            # making sure that connection is also updated for that preceding node
            node_list[i].suceeding_conns.append([self, DEFAULT_WEIGHT if weight_list is None else weight_list[i]])

        # don't forget to update value with the new connections
        if auto_update_values:
            self.value = self.calc_value()

    def connect_suceeding_nodes(self, node_list, weight_list: list[float] = None, auto_update_values = False):
        for i in range(len(node_list)):
            # adding new connections for suceeding nodes
            self.suceeding_conns.append([node_list[i], DEFAULT_WEIGHT if weight_list is None else weight_list[i]])

            # making sure that connection is also updated for that succeeding node
            node_list[i].preceding_conns.append([self, DEFAULT_WEIGHT if weight_list is None else weight_list[i]])

            # don't forget to update value of the suceeding node
            if auto_update_values:
                node_list[i].value = node_list[i].calc_value()
        
    def calc_value(self):
        summation = 0

        # going through each precedding connection and add up: value * weight
        for i in range(len(self.preceding_conns)):
            summation += self.preceding_conns[i][0].value * self.preceding_conns[i][1]

        # feed the summation into the activation function (don't forget to include the bias)
        return self.activation_function(summation + self.bias)
    


class Node_Layer:
    def __init__(
            self,
            node_list: list[Node] = None,
            alias: str = '',
            activation_function = lambda x: x,
            bias_list: list[float] = None,
        ):
        self.node_list = [] if node_list is None else node_list
        self.alias = alias

        # going through each node in node_list and setting their activation function & bias
        for i in range(len(node_list)):
            node_list[i].activation_function = activation_function

            node_list[i].bias = bias_list[i] if bias_list is not None else DEFAULT_BIAS

    def __str__(self):
        result = f'{self.alias}:\n'

        for i in range(len(self.node_list)):
            result += f'\t{str(self.node_list[i])}\n'

        return result
    
    def __repr__(self):
        result = f'Layer {self.alias} contain the following nodes:\n'

        for i in range(len(self.node_list)):
            result += f'\t{str(self.node_list[i])}'

        return result
    
    def connect_preceding_layer(self, prev_layer, weight_matrix: list[list[float]] = None, auto_update_values = False):
        for i in range(len(self.node_list)): # each row contains the weights for an object in this object's node_list
            self.node_list[i].connect_preceding_nodes(prev_layer.node_list, weight_list=weight_matrix[i] if weight_matrix is not None else None, auto_update_values=auto_update_values)

    def connect_suceeding_layer(self, suceeding_layer, weight_matrix: list[list[float]] = None, auto_update_values = False):
        for i in range(len(self.node_list)): # each row contains the weights for an object in this object's node_list
            self.node_list[i].connect_suceeding_nodes(suceeding_layer.node_list, weight_list=weight_matrix[i] if weight_matrix is not None else None, auto_update_values=auto_update_values)