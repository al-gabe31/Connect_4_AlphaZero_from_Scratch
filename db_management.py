# code for handling interactions with the Connect_4 database and all its tables

from neural_network import *
import datetime
import sqlite3

def store_neural_network(neural_network: Neural_Network, database_location: str, neural_network_verison: str = 'N/A'):
    # ==================== INSERTING INTO NEURAL_NETWORKS TABLE ====================
    neural_network_alias = neural_network.alias
    neural_network_creation_date = str(datetime.datetime.now())

    new_neural_network = [(neural_network_alias, neural_network_verison, neural_network_creation_date)]

    with sqlite3.connect(database_location) as conn:
        conn.execute('PRAGMA foreign_keys = ON;')
        cursor = conn.cursor()

        cursor.executemany(
            'INSERT INTO Neural_Networks (alias, version, creation_date) VALUES (?, ?, ?)',
            new_neural_network
        )



    # ==================== INSERTING INTO NEURAL_NETWORK_LAYERS ====================
    # getting the neural_network_id given alias & version
    sql_code = f"""
    SELECT neural_network_id
    FROM Neural_Networks
    WHERE
        alias = '{neural_network_alias}'
        AND version = '{neural_network_verison}'
    """

    neural_network_id = None

    with sqlite3.connect(database_location) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(sql_code)
        rows = cursor.fetchone()

        neural_network_id = rows['neural_network_id']

    # populating the list we'll be inserting into Neural_Network_Layers
    new_neural_network_layers = []

    # adding information for the input layer
    curr_layer = (neural_network_id, 0, len(neural_network.input_layer.node_list), 'input', neural_network.input_layer.node_list[0].activation_function.__name__)
    new_neural_network_layers.append(curr_layer)

    # adding information for the hidden layers
    for i in range(len(neural_network.hidden_layers)):
        curr_layer = (neural_network_id, i, len(neural_network.hidden_layers[i].node_list), 'hidden', neural_network.hidden_layers[i].node_list[0].activation_function.__name__)
        new_neural_network_layers.append(curr_layer)

    # adding information for the output layers
    for i in range(len(neural_network.output_layers)):
        curr_layer = (neural_network_id, i, len(neural_network.output_layers[i].node_list), 'output', neural_network.output_layers[i].node_list[0].activation_function.__name__)
        new_neural_network_layers.append(curr_layer)

    # finally inserting into the Neural_Network_Layers table
    with sqlite3.connect(database_location) as conn:
        conn.execute('PRAGMA foreign_keys = ON;')
        cursor = conn.cursor()

        cursor.executemany(
            'INSERT INTO Neural_Network_Layers (neural_network_id, layer_index, num_nodes, layer_type, activation_type) VALUES (?, ?, ?, ?, ?)',
            new_neural_network_layers
        )



    # ==================== INSERTING INTO LAYER_WEIGHTS ====================
    # getting the layer_id given the neural_network_id

    sql_code = f"""
    SELECT layer_id
    FROM Neural_Network_Layers
    WHERE 
        neural_network_id = {neural_network_id}
        AND layer_type IN ('hidden', 'output')
    ORDER BY 
        CASE
            WHEN layer_type = 'hidden' THEN 1
            WHEN layer_type = 'output' THEN 2
        END ASC,
        layer_index ASC
    """

    layer_ids = []

    with sqlite3.connect(database_location) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(sql_code)
        rows = cursor.fetchall()

        for i in range(len(rows)):
            layer_ids.append((rows[i]['layer_id'],))

    # populating the list we'll be inserting into Layer_Weights
    new_layer_weights = []

    # adding information for the hidden layer weights
    for i in range(len(neural_network.hidden_layers)):
        curr_weight_matrix = neural_network.hidden_layers[i].get_weights_matrix()

        for row_index in range(len(curr_weight_matrix)):
            for col_index in range(len(curr_weight_matrix[row_index])):
                curr_weights = (layer_ids[i][0], row_index, col_index, float(curr_weight_matrix[row_index][col_index]))
                new_layer_weights.append(curr_weights)

    # adding information for the output layer weights
    for i in range(len(neural_network.output_layers)):
        curr_weight_matrix = neural_network.output_layers[i].get_weights_matrix()

        for row_index in range(len(curr_weight_matrix)):
            for col_index in range(len(curr_weight_matrix[row_index])):
                curr_weights = (layer_ids[i+len(neural_network.hidden_layers)][0], row_index, col_index, float(curr_weight_matrix[row_index][col_index]))
                new_layer_weights.append(curr_weights)

    # finally inserting into the Layer_Weights table
    with sqlite3.connect(database_location) as conn:
        conn.execute('PRAGMA foreign_keys = ON;')
        cursor = conn.cursor()

        cursor.executemany(
            'INSERT INTO Layer_Weights (layer_id, row_num, col_num, weight_val) VALUES (?, ?, ?, ?)',
            new_layer_weights
        )

        

    # ==================== INSERTING INTO LAYER_BIASES ====================
    new_layer_biases = []

    # adding information for the hidden layer biases
    for i in range(len(neural_network.hidden_layers)):
        curr_bias_list = neural_network.hidden_layers[i].get_bias()

        for row_index in range(len(curr_bias_list)):
            curr_biases = (layer_ids[i][0], row_index, float(curr_bias_list[row_index]))
            new_layer_biases.append(curr_biases)

    # adding information for the output layer biases
    for i in range(len(neural_network.output_layers)):
        curr_bias_list = neural_network.output_layers[i].get_bias()

        for row_index in range(len(curr_bias_list)):
            curr_biases = (layer_ids[i+len(neural_network.hidden_layers)][0], row_index, float(curr_bias_list[row_index]))
            new_layer_biases.append(curr_biases)

    # finally inserting into the Layer_Biases table
    with sqlite3.connect(database_location) as conn:
        conn.execute('PRAGMA foreign_keys = ON;')
        cursor = conn.cursor()

        cursor.executemany(
            'INSERT INTO Layer_Biases (layer_id, node_num, bias_val) VALUES (?, ?, ?)',
            new_layer_biases
        )