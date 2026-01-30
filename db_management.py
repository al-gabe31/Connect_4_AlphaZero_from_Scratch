# code for handling interactions with the Connect_4 database and all its tables

from neural_network import *
import datetime
import sqlite3

def store_neural_network(
        neural_network: Neural_Network, 
        database_location: str, 
        neural_network_version: str = 'N/A'
    ):
    # ==================== INSERTING INTO NEURAL_NETWORKS TABLE ==================== #
    neural_network_alias = neural_network.alias
    neural_network_creation_date = str(datetime.datetime.now())

    new_neural_network = [(neural_network_alias, neural_network_version, neural_network_creation_date)]

    with sqlite3.connect(database_location) as conn:
        conn.execute('PRAGMA foreign_keys = ON;')
        cursor = conn.cursor()

        cursor.executemany(
            'INSERT INTO Neural_Networks (alias, version, creation_date) VALUES (?, ?, ?)',
            new_neural_network
        )



    # ==================== INSERTING INTO NEURAL_NETWORK_LAYERS ==================== #
    # getting the neural_network_id given alias & version
    sql_code = f"""
    SELECT neural_network_id
    FROM Neural_Networks
    WHERE
        alias = '{neural_network_alias}'
        AND version = '{neural_network_version}'
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



    # ==================== INSERTING INTO LAYER_WEIGHTS ==================== #
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

        

    # ==================== INSERTING INTO LAYER_BIASES ==================== #
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

def retrieve_neural_network(
        database_location: str,
        alias: str,
        version: str
    ):
    # ==================== RETRIEVING INFORMATION TO INITIALIZE NEURAL NETWORK ==================== #
    sql_code = f"""
    SELECT
        b.layer_id          AS layer_id,
        b.layer_index       AS layer_index,
        b.num_nodes         AS num_nodes,
        b.layer_type        AS layer_type,
        b.activation_type   AS activation_type
    FROM Neural_Networks AS a
    INNER JOIN Neural_Network_Layers AS b
        ON a.neural_network_id = b.neural_network_id
    WHERE
        a.alias = '{alias}'
        AND a.version = '{version}'
    ORDER BY
        CASE
            WHEN b.layer_type = 'input' THEN 1
            WHEN b.layer_type = 'hidden' THEN 2
            WHEN b.layer_type = 'output' THEN 3
        END ASC,
        b.layer_index ASC
    """

    query_result = []

    with sqlite3.connect(database_location) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(sql_code)
        rows = cursor.fetchall()

        for i in range(len(rows)):
            query_result.append((rows[i]['layer_id'], rows[i]['layer_index'], rows[i]['num_nodes'], rows[i]['layer_type'], rows[i]['activation_type']))

    # parsing the information
    input_layer_num_nodes:int = None
    hidden_layer_dimensions:list[int] = []
    output_layer_dimensions:list[int] = []
    activation_functions:list = []

    # going through the query_result list
    for i in range(len(query_result)):
        # information for the dimensions of each layer
        if query_result[i][3] == 'input':
            input_layer_num_nodes = query_result[i][2]
        elif query_result[i][3] == 'hidden':
            hidden_layer_dimensions.append(query_result[i][2])
        elif query_result[i][3] == 'output':
            output_layer_dimensions.append(query_result[i][2])

        # information for the activation functions for each layer
        if query_result[i][3] in ('hidden', 'output'):
            # we don't need to include the activation function for the input layer
            activation_functions.append(globals()[query_result[i][4]])

    # finally using the information to initialize the neural_network
    result_neural_network = Neural_Network(
        input_layer_num_nodes=input_layer_num_nodes,
        hidden_layer_dimensions=hidden_layer_dimensions,
        output_layer_dimensions=output_layer_dimensions,
        alias=alias,
        activation_functions=activation_functions
    )



    # ==================== RETRIEVING INFORMATION FOR THE WEIGHTS ==================== #
    weight_matrices = [] # contains weight matrices for all layers
    output_preceding_layer_index = 0 # starts with the output layer's preceding layer being the input layer
    for i in range(1, len(query_result)):
        curr_weight_matrix = None

        # get dimensions for the curr_weight_matrix
        if query_result[i][3] == 'hidden': # handles behavior for hidden layers
            output_preceding_layer_index = i
            curr_weight_matrix = np.zeros((query_result[i][2], query_result[i-1][2]))
        elif query_result[i][3] == 'output': # handles behavior for output layers
            curr_weight_matrix = np.zeros((query_result[i][2], query_result[output_preceding_layer_index][2]))

        weight_matrices.append(curr_weight_matrix)

    # getting the specific weights from the database
    sql_code_2 = f"""
    SELECT
        b.layer_type AS layer_type,
        b.layer_index AS layer_index,
        c.row_num AS row_num,
        c.col_num AS col_num,
        c.weight_val AS weight_val
    FROM Neural_Networks AS a
    INNER JOIN Neural_Network_Layers AS b
        ON a.neural_network_id = b.neural_network_id
    INNER JOIN Layer_Weights c
        ON b.layer_id = c.layer_id
    WHERE
        a.alias = '{alias}'
        AND a.version = '{version}'
        AND b.layer_type IN ('hidden', 'output')
    ORDER BY
        CASE
            WHEN b.layer_type = 'hidden' THEN 1
            WHEN b.layer_type = 'output' THEN 2
        END ASC,
        b.layer_index ASC,
        c.row_num ASC,
        c.col_num ASC
    """

    query_result_2 = []

    with sqlite3.connect(database_location) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(sql_code_2)
        rows = cursor.fetchall()

        for i in range(len(rows)):
            query_result_2.append((rows[i]['layer_type'], rows[i]['layer_index'], rows[i]['row_num'], rows[i]['col_num'], rows[i]['weight_val']))

    # updating weight matrices based on the query we just got
    hidden_layer_last_index = -1
    for i in range(len(query_result_2)):
        if query_result_2[i][0] == 'hidden':
            hidden_layer_last_index = query_result_2[i][1]
            weight_matrices[query_result_2[i][1]][query_result_2[i][2]][query_result_2[i][3]] = query_result_2[i][4]
        elif query_result_2[i][0] == 'output':
            weight_matrices[query_result_2[i][1]+hidden_layer_last_index+1][query_result_2[i][2]][query_result_2[i][3]] = query_result_2[i][4]

    # finally setting the weights for each layer in the neural network
    preceding_layer: Node_Layer = result_neural_network.input_layer
    weight_matrix_index = 0

    # updating weights for hidden layers
    for i in range(len(result_neural_network.hidden_layers)):
        result_neural_network.hidden_layers[i].update_preceding_weights(
            preceding_layer=preceding_layer,
            weight_matrix=weight_matrices[weight_matrix_index]
        )
        preceding_layer = result_neural_network.hidden_layers[i]
        weight_matrix_index += 1

    # updating weights for output layers
    for i in range(len(result_neural_network.output_layers)):
        preceding_col_start_index = sum([len(result_neural_network.output_layers[j].node_list) for j in range(0, i)])
        result_neural_network.output_layers[i].update_preceding_weights(
            preceding_layer=preceding_layer,
            weight_matrix=weight_matrices[weight_matrix_index],
            preceding_col_start_index=preceding_col_start_index
        )
        weight_matrix_index += 1



    # ==================== RETRIEVING INFORMATION FOR THE BIASES ==================== #
    bias_lists = [] # contains bias lists for all layers
    for i in range(1, len(query_result)):
        curr_bias_list = np.zeros(query_result[i][2])
        bias_lists.append(curr_bias_list)

    # getting the specific biases from the database
    sql_code_3 = f"""
    SELECT
        b.layer_type AS layer_type,
        b.layer_index AS layer_index,
        c.node_num AS node_num,
        c.bias_val AS bias_val
    FROM Neural_Networks AS a
    INNER JOIN Neural_Network_Layers AS b
        ON a.neural_network_id = b.neural_network_id
    INNER JOIN Layer_Biases c
        ON b.layer_id = c.layer_id
    WHERE
        a.alias = '{alias}'
        AND a.version = '{version}'
        AND b.layer_type IN ('hidden', 'output')
    ORDER BY
        CASE
            WHEN b.layer_type = 'hidden' THEN 1
            WHEN b.layer_type = 'output' THEN 2
        END ASC,
        b.layer_index ASC,
        c.node_num ASC
    """

    query_result_3 = []

    with sqlite3.connect(database_location) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(sql_code_3)
        rows = cursor.fetchall()

        for i in range(len(rows)):
            query_result_3.append((rows[i]['layer_type'], rows[i]['layer_index'], rows[i]['node_num'], rows[i]['bias_val']))

    # updating bias_lists based on the query we just got
    for i in range(len(query_result_3)):
        if query_result_3[i][0] == 'hidden':
            bias_lists[query_result_3[i][1]][query_result_3[i][2]] = query_result_3[i][3]
        elif query_result_3[i][0] == 'output':
            bias_lists[query_result_3[i][1]+len(result_neural_network.hidden_layers)][query_result_3[i][2]] = query_result_3[i][3]

    # finally setting the biases for each layer in the neural network
    bias_list_index = 0

    # updating biases for the hidden layers
    for i in range(len(result_neural_network.hidden_layers)):
        result_neural_network.hidden_layers[i].set_bias_list(bias_list=bias_lists[bias_list_index])
        bias_list_index += 1

    # updating biases for the output layers
    for i in range(len(result_neural_network.output_layers)):
        result_neural_network.output_layers[i].set_bias_list(bias_list=bias_lists[bias_list_index])
        bias_list_index += 1

    # don't forget to return the resulting neural network
    return result_neural_network