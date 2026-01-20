# contains all classes for a neural network from sctach

import math
import random

DEFAULT_WEIGHT = 1
DEFAULT_BIAS = 0

def default_activation(value):
    return value

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

def activation_derivative(func_name, value):
    if func_name == default_activation:
        return 1
    elif func_name == relu:
        return 1 if value > 0 else 0
    elif func_name == tanh:
        return 1 - pow(tanh(value), 2)
    elif func_name == sigmoid:
        return sigmoid(value) * (1 - sigmoid(value))
    else:
        return 0

# use for sigmoid
def xavier_normal(fan_in, fan_out):
    sigma = math.sqrt(2.0 / (fan_in + fan_out))
    return random.gauss(0.0, sigma)

# use for ReLu
def he_normal(fan_in, fan_out = None):
    # fan_out parameter won't be used
    # it's just there to make it flexible when deciding between using xavier or he
    sigma = math.sqrt(2.0 / fan_in)
    return random.gauss(0.0, sigma)

class Node:
    def __init__(
            self, 
            value: float = None,
            alias: str = '',
            preceding_conns = None,
            suceeding_conns = None,
            activation_function = default_activation,
            bias: float = None,
        ):
        self.value = value # value head of this node
        self.alias = alias # what this node will be called
        self.preceding_conns = [] if preceding_conns is None else preceding_conns # list of nodes that connect to this object and its corresponding weights
        self.suceeding_conns = [] if suceeding_conns is None else suceeding_conns # list of nodes this object connects to and its corresponding weights
        self.activation_function = activation_function # activation funciton to achieve non-linearity
        self.bias = DEFAULT_BIAS if bias is None else bias # bias term when calculating value head
        self.z_value: float = None
        self.delta_value: float = None # very important for backpropagation

    def __str__(self):
        return f'{self.alias}: {self.value} [z: {self.z_value} | delta: {self.delta_value}] ==> {[self.suceeding_conns[i][1] for i in range(len(self.suceeding_conns))]}'
    
    def __repr__(self):
        return f'{self.alias}: {self.value} [z: {self.z_value} | delta: {self.delta_value}] ==> {[self.suceeding_conns[i][1] for i in range(len(self.suceeding_conns))]}'

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
        
    # returns the activation value of a node (doesn't actually set it)
    def calc_value(self):
        return self.activation_function(self.get_z_value())
    
    def get_z_value(self):
        summation = 0

        # going through each preceding connection and add up: value * weight
        for i in range(len(self.preceding_conns)):
            summation += self.preceding_conns[i][0].value * self.preceding_conns[i][1]

        return summation + self.bias
    
    def get_delta_value(self, in_output_layer = False, y_value = None):
        if in_output_layer:
            return 2 * (self.value - y_value) * activation_derivative(self.activation_function, self.get_z_value())
        else:
            summation = 0

            # going through each node in suceeding connections: delta * corresponding weight
            for i in range(len(self.suceeding_conns)):
                summation += self.suceeding_conns[i][0].delta_value * self.suceeding_conns[i][1]

            # then multiply all that by the derivative of the local node's activation function
            summation *= activation_derivative(self.activation_function, self.get_z_value())

            return summation

    


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
        result = f'{self.alias}:\n'

        for i in range(len(self.node_list)):
            result += f'\t{str(self.node_list[i])}'

        return result
    
    def connect_preceding_layer(self, preceding_layer, weight_matrix: list[list[float]] = None, auto_update_values = False, weight_normalization = None, fan_in: int = None, fan_out: int = None):
        for i in range(len(self.node_list)): # each row contains the weights for an object in this object's node_list
            curr_weight_list = None

            # sets weight initialization
            if weight_normalization == xavier_normal:
                curr_weight_list = [xavier_normal(fan_in, fan_out) for i in range(len(preceding_layer.node_list))]
            elif weight_normalization == he_normal:
                curr_weight_list = [he_normal(fan_in) for i in range(len(preceding_layer.node_list))]
            elif weight_matrix is not None:
                # if no weight initializaiton is provided, use the provided weights instead
                curr_weight_list = weight_matrix[i]
            
            self.node_list[i].connect_preceding_nodes(preceding_layer.node_list, weight_list=curr_weight_list, auto_update_values=auto_update_values)

    def connect_suceeding_layer(self, suceeding_layer, weight_matrix: list[list[float]] = None, auto_update_values = False, weight_normalization = None, fan_in: int = None, fan_out: int = None):
        for i in range(len(self.node_list)): # each row contains the weights for an object in this object's node_list
            curr_weight_list = None

            # sets weight initialization
            if weight_normalization == xavier_normal:
                curr_weight_list = [xavier_normal(fan_in, fan_out) for i in range(len(suceeding_layer.node_list))]
            elif weight_normalization == he_normal:
                curr_weight_list = [he_normal(fan_in) for i in range(len(suceeding_layer.node_list))]
            elif weight_matrix is not None:
                # if no weight initialization is provided, use the provided weights instead
                curr_weight_list = weight_matrix[i]
            
            self.node_list[i].connect_suceeding_nodes(suceeding_layer.node_list, weight_list=curr_weight_list, auto_update_values=auto_update_values)

    # updates activation values for all nodes in this layer
    def calc_layer_values(self):
        for i in range(len(self.node_list)):
            self.node_list[i].value = self.node_list[i].calc_value()

    # updates z values for all nodes in this layer
    def calc_layer_z_values(self):
        for i in range(len(self.node_list)):
            self.node_list[i].z_value = self.node_list[i].get_z_value()

    # updates delta values for all nodes in this layer
    def calc_delta_values(self, is_output_layer = False, y_value = None):
        for i in range(len(self.node_list)):
            self.node_list[i].delta_value = self.node_list[i].get_delta_value(is_output_layer, y_value)