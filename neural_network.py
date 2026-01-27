# contains all classes for a neural network from sctach

import math
import random
import numpy as np

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

def softmax(value, values_list):
    return pow(math.e, value) / sum([pow(math.e, values_list[i]) for i in range(len(values_list))])

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

# returns a vector of size vector_size
# for every index stored in hit_list, set the value in the resulting vector at that index to hit_flag
def flag_setter(vector_size: int, hit_list: list[int], not_hit_flag: int = 0, hit_flag: int = 1):
    result = [not_hit_flag for i in range(vector_size)]

    for i in range(len(hit_list)):
        result[hit_list[i]] = hit_flag

    return result

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
        return f'{self.alias}: {self.value} [z: {self.z_value} | delta: {self.delta_value} | bias: {self.bias}] ==> {[self.suceeding_conns[i][1] for i in range(len(self.suceeding_conns))]}'
    
    def __repr__(self):
        return f'{self.alias}: {self.value} [z: {self.z_value} | delta: {self.delta_value} | bias: {self.bias}] ==> {[self.suceeding_conns[i][1] for i in range(len(self.suceeding_conns))]}'

    def connect_preceding_nodes(
            self, 
            node_list, 
            weight_list: list[float] = None, 
            auto_update_values = False, 
            layer_node_list = None
        ):
        for i in range(len(node_list)):
            # adding new connections for preceding nodes
            self.preceding_conns.append([node_list[i], DEFAULT_WEIGHT if weight_list is None else weight_list[i]])

            # making sure that connection is also updated for that preceding node
            node_list[i].suceeding_conns.append([self, DEFAULT_WEIGHT if weight_list is None else weight_list[i]])

        # don't forget to update value with the new connections
        if auto_update_values:
            self.value = self.calc_value(layer_node_list=layer_node_list)

    def connect_suceeding_nodes(
            self, 
            node_list, 
            weight_list: list[float] = None, 
            auto_update_values = False, 
            layer_node_list = None
        ):
        for i in range(len(node_list)):
            # adding new connections for suceeding nodes
            self.suceeding_conns.append([node_list[i], DEFAULT_WEIGHT if weight_list is None else weight_list[i]])

            # making sure that connection is also updated for that succeeding node
            node_list[i].preceding_conns.append([self, DEFAULT_WEIGHT if weight_list is None else weight_list[i]])

            # don't forget to update value of the suceeding node
            if auto_update_values:
                node_list[i].value = node_list[i].calc_value(layer_node_list=layer_node_list)
        
    # returns the activation value of a node (doesn't actually set it)
    def calc_value(
            self, 
            layer_node_list = None,
            masking_bits: list[int] = None, # example [0, 0, 1, 0, 1]
            is_masked: bool = False
        ):
        if self.activation_function == softmax:
            if masking_bits is None:
                masking_bits = flag_setter(vector_size=len(layer_node_list), hit_list=[], not_hit_flag=0, hit_flag=1)
            
            # activation value of this node relies on the z-values of the other nodes in the same layer
            # if a particular output node is marked as masked, set its z-value to negative infinity instead
            values_list = [layer_node_list[i].get_z_value() if masking_bits[i] == 0 else -math.inf for i in range(len(layer_node_list))]

            return self.activation_function(self.get_z_value() if not is_masked else -math.inf, values_list)
        else:
            return self.activation_function(self.get_z_value())
    
    def get_z_value(self):
        summation = 0

        # going through each preceding connection and add up: value * weight
        for i in range(len(self.preceding_conns)):
            summation += self.preceding_conns[i][0].value * self.preceding_conns[i][1]

        return summation + self.bias
    
    def get_delta_value(
            self, 
            in_output_layer: bool = False, 
            y_value: float = None
        ):
        if in_output_layer:
            if self.activation_function == softmax:
                return self.value - y_value # activation - actual
            else:
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
            activation_function = default_activation,
            bias_list: list[float] = None,
        ):
        self.node_list = [] if node_list is None else node_list
        self.alias = alias

        # going through each node in node_list and setting their activation function & bias
        for i in range(len(node_list)):
            node_list[i].activation_function = activation_function

            node_list[i].bias = bias_list[i] if bias_list is not None else DEFAULT_BIAS

    def __str__(self):
        activation_name = '' if self.node_list[0].activation_function == default_activation else f' [{self.node_list[0].activation_function.__name__}]'
        result = f'{self.alias}{activation_name}:\n'

        for i in range(len(self.node_list)):
            result += f'\t{str(self.node_list[i])}\n'

        return result
    
    def __repr__(self):
        activation_name = '' if self.node_list[0].activation_function == default_activation else f' [{self.node_list[0].activation_function.__name__}]'
        result = f'{self.alias}{activation_name}:\n'

        for i in range(len(self.node_list)):
            result += f'\t{str(self.node_list[i])}'

        return result
    
    def connect_preceding_layer(
            self, 
            preceding_layer, 
            weight_matrix: list[list[float]] = None, 
            auto_update_values = False, 
            weight_normalization = None, 
            fan_in: int = None, 
            fan_out: int = None
        ):
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
            
            self.node_list[i].connect_preceding_nodes(preceding_layer.node_list, weight_list=curr_weight_list, auto_update_values=auto_update_values, layer_node_list=self.node_list)

    def update_preceding_weights(
            self, 
            preceding_layer, 
            weight_matrix: list[list[float]] = None,
            preceding_col_start_index: int = 0, # useful for multitask learning where the preceding layer connects to multiple suceeding layers
        ):
        for row_index in range(len(weight_matrix)):
            for col_index in range(len(weight_matrix[row_index])):
                self.node_list[row_index].preceding_conns[col_index][1] = weight_matrix[row_index][col_index]

                preceding_layer.node_list[col_index].suceeding_conns[row_index + preceding_col_start_index][1] = weight_matrix[row_index][col_index]

    def connect_suceeding_layer(
            self, 
            suceeding_layer, 
            weight_matrix: list[list[float]] = None, 
            auto_update_values = False, 
            weight_normalization = None, 
            fan_in: int = None, 
            fan_out: int = None
        ):
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
            
            self.node_list[i].connect_suceeding_nodes(suceeding_layer.node_list, weight_list=curr_weight_list, auto_update_values=auto_update_values, layer_node_list=self.node_list)

    # updates activation values for all nodes in this layer
    def calc_layer_values(
            self,
            masking_bits: list[int] = None # example [0, 0, 1, 0, 1]
        ):
        for i in range(len(self.node_list)):
            if masking_bits is None:
                self.node_list[i].value = self.node_list[i].calc_value(layer_node_list=self.node_list)
            else:
                is_masked = True if masking_bits[i] == 1 else False
                self.node_list[i].value = self.node_list[i].calc_value(layer_node_list=self.node_list, masking_bits=masking_bits, is_masked=is_masked)

    # updates z values for all nodes in this layer
    def calc_layer_z_values(self):
        for i in range(len(self.node_list)):
            self.node_list[i].z_value = self.node_list[i].get_z_value()

    # updates delta values for all nodes in this layer
    def calc_delta_values(
            self, 
            is_output_layer: bool = False, 
            y_values: list[float] = None
        ):
        if not is_output_layer:
            for i in range(len(self.node_list)):
                self.node_list[i].delta_value = self.node_list[i].get_delta_value()
        else:
            for i in range(len(self.node_list)):
                self.node_list[i].delta_value = self.node_list[i].get_delta_value(in_output_layer=True, y_value=y_values[i])

    # sets activation, z-value, and delta to 0 for all nodes in this layer
    def clear_layer_data(self):
        for i in range(len(self.node_list)):
            self.node_list[i].value = None
            self.node_list[i].z_value = None
            self.node_list[i].delta_value = None

    def set_activation_function(
            self, 
            activation_function
        ):
        for i in range(len(self.node_list)):
            self.node_list[i].activation_function = activation_function

    def set_bias_list(
            self, 
            bias_list: list[float]
        ):
        for i in range(len(self.node_list)):
            self.node_list[i].bias = bias_list[i]

    # get's the weights matrix of this layer of dimensions nxm
        # n: number of nodes in this layer
        # m: number of nodes in the preceding layer
    def get_weights_matrix(self):
        weights_matrix = []

        for i in range(len(self.node_list)):
            curr_row_weights = []
            for j in range(len(self.node_list[i].preceding_conns)):
                curr_row_weights.append(self.node_list[i].preceding_conns[j][1])

            curr_row_weights = np.array(curr_row_weights)
            weights_matrix.append(curr_row_weights)
        weights_matrix = np.array(weights_matrix)

        return weights_matrix
    
    # get's the bias list of this layer
    def get_bias(self):
        bias_list = []

        for i in range(len(self.node_list)):
            bias_list.append(self.node_list[i].bias)
        bias_list = np.array(bias_list)

        return bias_list



class Neural_Network:
    def __init__(
            self,
            input_layer_num_nodes: int,
            hidden_layer_dimensions: list[int],
            output_layer_dimensions: list[int], # multiple output layers supports multitask learning
            alias: str = '',
            activation_functions = None, # list of activation functions (length should match # of layers - 1)
            weight_normalizations = None, # list of weight normalizations (length should match # of layers - 1)
            bias_lists: list[list[float]] = None
        ):

        self.alias = alias
        self.fan_in = input_layer_num_nodes # number of input nodes
        self.fan_out = sum(output_layer_dimensions) # number of output nodes

        # initializing the input, hidden, and output layers
        self.input_layer: Node_Layer = Node_Layer(
            [Node(alias=f'n1{i + 1}') for i in range(input_layer_num_nodes)],
            alias='INPUT l1'
        )

        self.hidden_layers: list[Node_Layer] = []
        for i in range(len(hidden_layer_dimensions)):
            new_hidden_layer = Node_Layer(
                [Node(alias=f'n{i + 2}{j + 1}') for j in range(hidden_layer_dimensions[i])],
                alias=f'HIDDEN l{i + 2}'
            )

            self.hidden_layers.append(new_hidden_layer)

        self.output_layers: list[Node_Layer] = []
        for i in range(len(output_layer_dimensions)):
            new_output_layer = Node_Layer(
                [Node(alias=f'n{i + len(hidden_layer_dimensions) + 2}{j + 1}') for j in range(output_layer_dimensions[i])],
                alias=f'OUTPUT l{i + len(hidden_layer_dimensions) + 2}'
            )

            self.output_layers.append(new_output_layer)

        # setting up the activation functions for each layer
        if activation_functions is None:
            activation_functions = [default_activation for i in range(len(self.hidden_layers) + 1)] # just use the default activation function for all layers that isn't the input layer
        
        for i in range(len(activation_functions)):
            if i < len(self.hidden_layers): # setting activation functions for the hidden layers
                self.hidden_layers[i].set_activation_function(activation_functions[i])
            else: # setting activation function for the output layers
                self.output_layers[i - len(hidden_layer_dimensions)].set_activation_function(activation_functions[i])
            # any overflow will be ignored

        # setting up the connections layer by layer
        if weight_normalizations is None:
            # handles situations where weight_normalization is passed as None
            weight_normalizations = [None for i in range(len(self.hidden_layers) + len(self.output_layers))] # just fill weight normalizations with None (will just set all weights to 1)
        for i in range(len(self.hidden_layers) + len(self.output_layers) - len(weight_normalizations)): # handles situations where there aren't enough weight normalizations in the list
            weight_normalizations.append(None) # adds any missing weight normalizations as None

        for i in range(len(weight_normalizations)):
            if i == 0 and len(self.hidden_layers) != 0: # input layer to the first hidden layer (assuming it exists)
                self.input_layer.connect_suceeding_layer(self.hidden_layers[i], weight_normalization=weight_normalizations[i], fan_in=self.fan_in, fan_out=self.fan_out)
            elif len(self.hidden_layers) == 0: # input layer to output layers (there are no hidden layers)
                self.input_layer.connect_suceeding_layer(self.output_layers[i], weight_normalization=weight_normalizations[i], fan_in=self.fan_in, fan_out=self.fan_out)
            elif i < len(self.hidden_layers): # hidden layer to hidden layer
                self.hidden_layers[i - 1].connect_suceeding_layer(self.hidden_layers[i], weight_normalization=weight_normalizations[i], fan_in=self.fan_in, fan_out=self.fan_out)
            elif i >= len(self.hidden_layers) and i < len(self.hidden_layers) + len(self.output_layers): # last hidden layer to output layers
                self.hidden_layers[-1].connect_suceeding_layer(self.output_layers[i - len(self.hidden_layers)], weight_normalization=weight_normalizations[i], fan_in=self.fan_in, fan_out=self.fan_out)

        # setting up the biases for each layer that isn't the input layer
        if bias_lists is None: # if bias_lists is None, just set all biases to DEFAULT_BIAS
            bias_lists = []

        # populate bias_lists to match the number of hidden & output layers
        for i in range(len(self.hidden_layers)):
            if i >= len(bias_lists): # not enough biases for the hidden layers
                bias_lists.append([DEFAULT_BIAS for j in range(len(self.hidden_layers[i].node_list))])

        for i in range(len(self.hidden_layers), len(self.hidden_layers) + len(self.output_layers)):
            if i >= len(bias_lists): # not enough biases for the output layers
                bias_lists.append([DEFAULT_BIAS for j in range(len(self.output_layers[i - len(self.hidden_layers)].node_list))])
        
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i].set_bias_list(bias_lists[i])
        for i in range(len(self.output_layers)):
            self.output_layers[i].set_bias_list(bias_lists[i + len(self.hidden_layers)])

    def __str__(self):
        result = f'{self.alias}:\n'
        result += f'{str(self.input_layer)}\n'

        for i in range(len(self.hidden_layers)):
            result += f'{str(self.hidden_layers[i])}\n'

        for i in range(len(self.output_layers)):
            result += f'{str(self.output_layers[i])}\n'

        return result
    
    def __repr__(self):
        result = f'{self.alias}:\n'
        result += f'{str(self.input_layer)}\n'

        for i in range(len(self.hidden_layers)):
            result += f'{str(self.hidden_layers[i])}\n'

        for i in range(len(self.output_layers)):
            result += f'{str(self.output_layers[i])}\n'

        return result
    
    def input_values(
            self, 
            input_set: list[float]
        ):
        # initializing inputs in the input layer
        for i in range(len(input_set)):
            self.input_layer.node_list[i].value = input_set[i]

    def forward_pass(
            self,
            masking_setting: list[list[int]] = None # example [[1, 0], [[0, 0, 1, 0, 1], [0]]]
        ):
        # forward pass hidden layers
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i].calc_layer_values()
            self.hidden_layers[i].calc_layer_z_values()

        # forward pass output layers
        for i in range(len(self.output_layers)):
            if masking_setting is None:
                self.output_layers[i].calc_layer_values()
            else:
                # if the layer is masked as idenfitifed by the i-th index of the 1st list in masking_setting...
                # take the i-th list from the 2nd list in masking_setting and set that as the masking_bits
                layer_is_masked = True if masking_setting[0][i] == 1 else False
                self.output_layers[i].calc_layer_values(masking_bits=None if not layer_is_masked else masking_setting[1][i])
            self.output_layers[i].calc_layer_z_values()

    def backwardpass(
            self, 
            y_values: list[list[float]] # each inner list represents an output set for multitask learning
        ):
        # calculate delta values for output layers
        for i in range(len(y_values)): # splits output sets and plugs them into their corresponding output layer
            self.output_layers[i].calc_delta_values(is_output_layer=True, y_values=y_values[i])

        # calculating delta values for the hidden layers from right to left
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            self.hidden_layers[i].calc_delta_values()

    def backpropagation_weights(self):
        # 3D matrix that contains all partial derivatives with respect to a specific weight
        # dimensions: l x n x m
            # l: number of hidden + output layers
            # n: number of nodes in that layer
            # m: number of preceding connections that node has
        weight_partial_derivatives = []

        # getting weight partial derivatives starting from the output layers
        for output_layer_index in range(len(self.output_layers) - 1, -1, -1): # "for each output layer..." (starting from the "last" output layer)
            # initializing dimensions of current weight partials matrix
            n_nodes = len(self.output_layers[output_layer_index].node_list) # number of nodes in an output layer
            m_nodes = len(self.input_layer.node_list) if len(self.hidden_layers) == 0 else len(self.hidden_layers[-1].node_list) # number of nodes in its preceding layer (either input or last hidden layer)
            curr_weight_partials = np.zeros((n_nodes, m_nodes)) # dimension: n_nodes x m_nodes
            
            for i in range(len(self.output_layers[output_layer_index].node_list)): # "for each node in that output layer..."
                for j in range(len(self.output_layers[output_layer_index].node_list[i].preceding_conns)): # "for each preceding connection of that node..."
                    # calculating partial derivative with respect to a specific weight: a_L-1 * delta_L
                    curr_weight_partials[i][j] = self.output_layers[output_layer_index].node_list[i].preceding_conns[j][0].value * self.output_layers[output_layer_index].node_list[i].delta_value

            weight_partial_derivatives.insert(0, curr_weight_partials)
        
        # doing the same for the hidden layer (besides the one after the input layer)
        for layer_index in range(len(self.hidden_layers) - 1, 0, -1): # stops at the left-most hidden layer (doesn't iterate past it)
            n_nodes = len(self.hidden_layers[layer_index].node_list) # number of nodes in that hidden layer
            m_nodes = len(self.hidden_layers[layer_index-1].node_list) # number of nodes in its preceding layer
            curr_weight_partials = np.zeros((n_nodes, m_nodes))
            
            for i in range(len(self.hidden_layers[layer_index].node_list)):
                for j in range(len(self.hidden_layers[layer_index].node_list[i].preceding_conns)):
                    # calculating partial derivative with respect to a specific weight: a_L-1 * delta_L 
                    curr_weight_partials[i][j] = self.hidden_layers[layer_index].node_list[i].preceding_conns[j][0].value * self.hidden_layers[layer_index].node_list[i].delta_value
            weight_partial_derivatives.insert(0, curr_weight_partials)

        # finally getting the partial derivaties from the left-most hidden layer
        if len(self.hidden_layers) >= 1: # first have to check that there even is a hidden layer
            n_nodes = len(self.hidden_layers[0].node_list) # number of nodes in the first hidden layer
            m_nodes = len(self.input_layer.node_list) # number of nodes in the input layer
            curr_weight_partials = np.zeros((n_nodes, m_nodes))

            for i in range(len(self.hidden_layers[0].node_list)):
                for j in range(len(self.hidden_layers[0].node_list[i].preceding_conns)):
                    curr_weight_partials[i][j] = self.hidden_layers[0].node_list[i].preceding_conns[j][0].value * self.hidden_layers[0].node_list[i].delta_value
            weight_partial_derivatives.insert(0, curr_weight_partials)

        return weight_partial_derivatives
    
    def backproagation_biases(self):
        # 2D matrix that contains all partial derivatives with respect to a specific bias
        # dimensions: l x n
            # l: number of hidden + output layers
            # n: number of nodes in that layer
        bias_partial_derivatives = []

        # getting bias partial derivatives starting from the output layers
        for output_layer_index in range(len(self.output_layers) -1, -1, -1): # "for each output layer..." (starting from the "last" output layer)
            curr_bias_partials = np.zeros(len(self.output_layers[output_layer_index].node_list))

            for i in range(len(self.output_layers[output_layer_index].node_list)): # "for each node in that output layer..."
                curr_bias_partials[i] = self.output_layers[output_layer_index].node_list[i].delta_value
            bias_partial_derivatives.insert(0, curr_bias_partials)

        # doing the same for the hidden layers
        for layer_index in range(len(self.hidden_layers) - 1, -1, -1): # goes through each hidden layer from right to left
            curr_bias_partials = np.zeros(len(self.hidden_layers[layer_index].node_list))

            for i in range(len(self.hidden_layers[layer_index].node_list)):
                curr_bias_partials[i] = self.hidden_layers[layer_index].node_list[i].delta_value
            bias_partial_derivatives.insert(0, curr_bias_partials)

        return bias_partial_derivatives
    
    def clear_neural_network(self):
        # clearing output layers
        for i in range(len(self.output_layers) - 1, -1, -1):
            self.output_layers[i].clear_layer_data()

        # clearing hidden layers
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            self.hidden_layers[i].clear_layer_data()

        # clearing input layers
        self.input_layer.clear_layer_data()

    def learn_data(
            self, 
            input_list: list[list[float]], 
            expected_list: list[list[list[float]]], 
            learning_rate: float = 0.001, 
            regularization: str = '',
            lambda_const: int = 1, 
        ):
        n = len(input_list) # number of input sets
        weight_partials = [] # will be used to update weights
        bias_partials = [] # will be used to update biases

        # run forward and backpropagation for each input set and expected output sets
        for i in range(len(input_list)):
            # 1. inserts input set into the input layer
            self.input_values(input_list[i])

            # 2. forward pass data down the neural network
            self.forward_pass()

            # 3. calculate delta values downstream using the expected output
            self.backwardpass(expected_list[i])

            # 4. tally partial derivatives for weights and biases
            curr_weight_partials = self.backpropagation_weights()
            curr_bias_partials = self.backproagation_biases()

            # if weight and bias partials list is currently empty, just append them to their respective arrays
            if i == 0:
                for j in range(len(curr_weight_partials)):
                    weight_partials.append(curr_weight_partials[j])
                for j in range(len(curr_bias_partials)):
                    bias_partials.append(curr_bias_partials[j])
            # otherwise, just tally (add) them to the populated partial ararys
            else:
                for j in range(len(curr_weight_partials)):
                    weight_partials[j] += curr_weight_partials[j]
                for j in range(len(curr_bias_partials)):
                    bias_partials[j] += curr_bias_partials[j]

            # 5. clear the neural network before moving on to the next input set
            self.clear_neural_network()

        # 6. averages the tallies in the weight and bias partials
        for i in range(len(weight_partials)):
            weight_partials[i] /= n
        for i in range(len(bias_partials)):
            bias_partials[i] /= n

        # 7. updates the weights and biases
        # updating the weights below
        for i in range(len(self.hidden_layers) + len(self.output_layers)):
            # getting regularization term
            reg_penalty = 0
            
            if regularization.upper() == 'L1' or regularization.upper() == 'LASSO': # lasso regression
                reg_penalty = learning_rate * lambda_const * (np.sign(self.hidden_layers[i].get_weights_matrix() if i < len(self.hidden_layers) else self.output_layers[i - len(self.hidden_layers)].get_weights_matrix()))
            elif regularization.upper() == 'L2' or regularization.upper() == 'RIDGE': # ridge regression
                reg_penalty = 2 * learning_rate * lambda_const * (self.hidden_layers[i].get_weights_matrix() if i < len(self.hidden_layers) else self.output_layers[i - len(self.hidden_layers)].get_weights_matrix())
            elif regularization.upper() == 'ELASTIC NET':
                reg_penalty = (2 * learning_rate * lambda_const * (self.hidden_layers[i].get_weights_matrix() if i < len(self.hidden_layers) else self.output_layers[i - len(self.hidden_layers)].get_weights_matrix())) + (learning_rate * lambda_const * (np.sign(self.hidden_layers[i].get_weights_matrix() if i < len(self.hidden_layers) else self.output_layers[i - len(self.hidden_layers)].get_weights_matrix())))
            
            # updating weights in hidden layers
            if i < len(self.hidden_layers):
                new_weights = self.hidden_layers[i].get_weights_matrix() - (learning_rate * weight_partials[i]) - reg_penalty

                if i == 0: # first hidden layer (its preceding layer is the input layer)
                    self.hidden_layers[i].update_preceding_weights(self.input_layer, new_weights)
                else: # otherwise, the preceding layer will just be another hidden layer
                    self.hidden_layers[i].update_preceding_weights(self.hidden_layers[i-1], new_weights)

            # updating the weights in the output layers
            elif i >= len(self.hidden_layers) and i < len(self.hidden_layers) + len(self.output_layers):
                new_weights = self.output_layers[i - len(self.hidden_layers)].get_weights_matrix() - (learning_rate * weight_partials[i]) - reg_penalty
                preceding_col_start_index = sum([len(self.output_layers[layer_index].node_list) for layer_index in range(0, i - len(self.hidden_layers))])

                # if there are hidden layers, the output layers' preceding layer will be a hidden layer
                if len(self.hidden_layers) > 0:
                    self.output_layers[i - len(self.hidden_layers)].update_preceding_weights(self.hidden_layers[-1], new_weights, preceding_col_start_index=preceding_col_start_index)
                # otherwise, the output layers' preceding layer will be the input layer
                else:
                    self.output_layers[i - len(self.hidden_layers)].update_preceding_weights(self.input_layer, new_weights, preceding_col_start_index=preceding_col_start_index)

        # updating the biases below
        for i in range(len(self.hidden_layers) + len(self.output_layers)):
            # updating the biases in the hidden layers
            if i < len(self.hidden_layers):
                new_biases = self.hidden_layers[i].get_bias() - (learning_rate * bias_partials[i])

                for j in range(len(self.hidden_layers[i].node_list)):
                    self.hidden_layers[i].node_list[j].bias = new_biases[j]
            # updating the biases in the output layers
            elif i >= len(self.hidden_layers) and i < len(self.hidden_layers) + len(self.output_layers):
                new_biases = self.output_layers[i - len(self.hidden_layers)].get_bias() - (learning_rate * bias_partials[i])

                for j in range(len(self.output_layers[i - len(self.hidden_layers)].node_list)):
                    self.output_layers[i - len(self.hidden_layers)].node_list[j].bias = new_biases[j]

    def get_output(
            self, 
            input_set: list[float],
            masking_setting: list[list[int]] = None # example [[1, 0], [[0, 0, 1, 0, 1], [0]]]
        ):
        self.input_values(input_set)
        self.forward_pass(masking_setting=masking_setting)

        outputs = [] # 2D list of output sets for multitask learning
        for layer_index in range(len(self.output_layers)):
            curr_layer_output = [] # current output set for a specific output layer
            
            for node_index in range(len(self.output_layers[layer_index].node_list)):
                curr_layer_output.append(self.output_layers[layer_index].node_list[node_index].value)
            curr_layer_output = np.array(curr_layer_output)
                
            outputs.append(curr_layer_output)
        
        return outputs
    
    def multi_run_learn_data(
            self, 
            input_list: list[list[float]], 
            expected_list: list[list[list[float]]], 
            learning_rate: float = 0.001, 
            epochs: int = 1000,
            regularization: str = '',
            lambda_const: int = 1,
            sampling_rate: float = 1,
        ):

        # implementation for mini-batch gradient descent
        n = len(input_list) # total number of inputs
        combined = [[input_list[i], expected_list[i]] for i in range(n)] # combining input & expected into a single list
        n_samples = math.floor(sampling_rate * n) # the number of entries per sample
        
        for epoch in range(epochs):
            # get new sample
            sampled = random.sample(combined, n_samples)
            input_sampled = [sampled[i][0] for i in range(n_samples)]
            expected_sampled = [sampled[i][1] for i in range(n_samples)]
            
            self.learn_data(
                input_list=input_sampled, 
                expected_list=expected_sampled, 
                learning_rate=learning_rate,
                regularization=regularization,
                lambda_const=lambda_const
            )