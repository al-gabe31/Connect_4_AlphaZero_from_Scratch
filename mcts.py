# implementations for monte carlo tree search here

from neural_network import *
from game_repr import *

class MCTS_Node:
    def __init__(
            self,
            game_state: Game_State,
            parent_node,
            neural_network: Neural_Network # assumes neural network is dual headed with the policy head being the 1st output layer and value head being the 2nd output layer
        ):
        # information about this node
        self.game_state: Game_State = game_state
        self.num_visits: int = 0 # how many times this node has been visited
        self.value_head: float = None # [1, -1] where 1 means winning state, -1 losing state, 0 even state
        self.policy_head: list[float] = None # guides expansion
        self.total_value: float = 0 # accumulation of all value heads that pass through this node during backpropagation
        self.mean_value: float = 0 # total_value / num_visits
        self.neural_network: Neural_Network = neural_network # used to calculate value & policy head

        # information about relationships with other nodes
        self.parent_node = parent_node # None for the root node
        self.children_connections = [None for i in range(NUM_COLS)] # list of MCTS_Node where the index specifies the specific move to get to that state
        self.expandable_flags = [1 if self.game_state.is_valid_move(i) else 0 for i in range(NUM_COLS)] # list of 1's and 0's where a 1 means a valid move that hasn't been added to the tree yet. if it's all 0's, then this node is fully expanded.

    def __str__(self):
        turn_number = self.game_state.get_turn_number()
        move_history = self.game_state.game_history
        
        result = '='*80 + '\n'
        result += f'Turn #{turn_number}\n'
        result += f'History: {move_history}\n'
        result += f'Num Visits: {self.num_visits}\n'
        result += f'Total Value: {self.total_value}\n'
        result += f'Mean Value: {self.mean_value}\n'
        result += f'Value Head: {self.value_head}\n'
        result += f'Policy Head: {self.policy_head}\n'
        result += f'Expandable Flags: {self.expandable_flags}\n'

        result += '\n'
        result += 'Board Grid\n'
        result += str(self.game_state)
        result += '\n'

        result += '='*80 + '\n'

        return result
    
    def __repr__(self):
        turn_number = self.game_state.get_turn_number()
        move_history = self.game_state.game_history
        
        result = '='*80 + '\n'
        result += f'Turn #{turn_number}\n'
        result += f'History: {move_history}\n'
        result += f'Num Visits: {self.num_visits}\n'
        result += f'Total Value: {self.total_value}\n'
        result += f'Mean Value: {self.mean_value}\n'
        result += f'Value Head: {self.value_head}\n'
        result += f'Policy Head: {self.policy_head}\n'
        result += f'Expandable Flags: {self.expandable_flags}\n'

        result += '\n'
        result += 'Board Grid\n'
        result += str(self.game_state)
        result += '\n'

        result += '='*80 + '\n'

        return result

    # uses PUCT to find the child node that balances exploration and exploitation
    # returns the selected node or None if it has no children
    def selection(
            self,
            exploration_constant: float = 1
        ):
        if all(child is None for child in self.children_connections):
            return None # ideally you shouldn't be calling selection on a terminal state
        
        puct_scores = []

        # calculates PUCT scores for each child connection
        # if a "child" is None, its PUCT score will be negative infinity
        for i in range(len(self.children_connections)):
            if self.children_connections[i] is None:
                puct_scores.append(-math.inf)
            else:
                puct_score = self.children_connections[i].mean_value + (exploration_constant * self.policy_head[i] * (math.sqrt(self.num_visits) / (1 + self.children_connections[i].num_visits)))
                puct_scores.append(puct_score)

        # choose child with the greatest PUCT score, increment its visits, and return it
        greatest_score = max(puct_scores)
        greatest_score_index = puct_scores.index(greatest_score)
        selected_child = self.children_connections[greatest_score_index]
        selected_child.num_visits += 1
        return selected_child
    
    # uses policy head and epandable flags to determine which unexpanded node will be added to the tree
    # returns the expanded (or added) node
    # returns None if using expansion on a fully expanded node
    def expansion(self):
        if all(flag == 0 for flag in self.expandable_flags):
            return None # only happens when using expansion on a fully expanded node
        
        expansion_scores = [self.policy_head[i] * self.expandable_flags[i] for i in range(len(self.expandable_flags))]
        greatest_score = max(expansion_scores)
        greatest_score_index = expansion_scores.index(greatest_score)

        # creates the new MCTS_Node object
        new_game_state = self.game_state.make_move(greatest_score_index)
        new_node = MCTS_Node(
            game_state=new_game_state,
            parent_node=self,
            neural_network=self.neural_network,
        )

        # updates information for this node
        self.children_connections[greatest_score_index] = new_node # sets connection to that new node
        self.expandable_flags[greatest_score_index] = 0 # meaning the node for that move has already been expanded (added)

        return new_node
    
    # calculate value and policy head for this node
    def evaluation(self):
        # if used on a game state where the game is over...
        # do nothing
        if self.game_state.game_over == True:
            return # ideally you shouldn't be calling evaluation on a terminal node
        
        # passing information about this node's game state into the neural network
        nn_input = self.game_state.one_hot_state_encoding()
        masking_setting = [[1, 0], [[0 if self.game_state.is_valid_move(i) else 1 for i in range(NUM_COLS)], [0]]]
        self.policy_head, self.value_head = self.neural_network.get_output(
            input_set=nn_input,
            masking_setting=masking_setting
        )

        self.value_head = self.value_head[0] # turning the list into a scalar

    # updates total & mean value based on passed value
    # calls backpropagation onto parent node (if it exists)
    def backpropagation(
            self,
            value: float,
        ):
        self.total_value += value
        self.mean_value = self.total_value / self.num_visits

        if self.parent_node is not None:
            self.parent_node.backpropagation(value=-1 * value) # don't forget to switch sign when moving up to parent