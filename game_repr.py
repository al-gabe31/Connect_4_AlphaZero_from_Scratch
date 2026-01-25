# everything we'll need to represent game states

import numpy as np
import copy

NUM_ROWS = 6
NUM_COLS = 7

class Game_State:
    def __init__(
            self,
            game_history: list[int] = None,
            outcome: int = 0,
            game_over: bool = False,
        ):
        
        # setting attributes
        self.game_history = [] if game_history is None else game_history
        self.outcome = outcome # 1 = player 1 wins | -1 = player 2 wins | 0 = no winner
        self.game_over = game_over

        # builds the grid for more convenient game representation
        self.grid: list[list[int]] = [[0 for i in range(NUM_COLS)] for j in range(NUM_ROWS)]
        curr_player = 1 # 1 = player 1 | -1 = player 2
        col_bottom_indexes = [5 for i in range(NUM_COLS)] # quick indexes for the bottom of a column

        # go through each turn in game_history and places a piece at the bottom of the column
        for column_placement in self.game_history:
            self.grid[col_bottom_indexes[column_placement]][column_placement] = curr_player # places piece at the bottom of the grid
            col_bottom_indexes[column_placement] -= 1 # raises the bottom of the column (reminder: row 0 is the top row)
            curr_player *= -1 # switches players

    def __str__(self):
        result = ''

        for i in range(NUM_ROWS):
            curr_row = '['
            for j in range(NUM_COLS - 1):
                curr_row += f'{' 0' if self.grid[i][j] == 0 else ' 1' if self.grid[i][j] == 1 else '-1'}, '
            curr_row += f'{' 0' if self.grid[i][-1] == 0 else ' 1' if self.grid[i][-1] == 1 else '-1'}, '
            

            curr_row += ']\n'
            result += curr_row
        result += f'outcome: {self.outcome}\n'
        result += f'game_over: {self.game_over}'

        return result
    
    def __repr__(self):
        result = ''

        for i in range(NUM_ROWS):
            curr_row = '['
            for j in range(NUM_COLS - 1):
                curr_row += f'{' 0' if self.grid[i][j] == 0 else ' 1' if self.grid[i][j] == 1 else '-1'}, '
            curr_row += f'{' 0' if self.grid[i][-1] == 0 else ' 1' if self.grid[i][-1] == 1 else '-1'}, '
            

            curr_row += ']\n'
            result += curr_row
        result += f'outcome: {self.outcome}\n'
        result += f'game_over: {self.game_over}'

        return result
    
    def get_grid_str(self):
        grid_result: list[list[int]] = []

        for row in self.grid:
            curr_row = [' 0' if row[col_index] == 0 else ' 1' if row[col_index] == 1 else '-1' for col_index in range(len(row))]
            grid_result.append(curr_row)

        return grid_result
    
    def is_valid_move(self, move: int):
        if self.outcome != 0: return False # game is already over
        if self.game_over: return False # game is already over
        if not isinstance(move, int): return False # move must be int
        if not (0 <= move and move < NUM_COLS): return False # move must in the range [0, 6]
        if self.grid[0][move] != 0: return False # column is full
        
        return True # passed all checks
    
    def get_player(self):
        return 1 if len(self.game_history) % 2 == 0 else -1
    
    def get_turn_number(self):
        return len(self.game_history) + 1
    
    def make_move(self, move: int):
        # first check that the move is valid
        if not self.is_valid_move(move):
            raise ValueError(f'EXCEPTION: INVALID MOVE {move}')
        
        new_game_history = self.game_history[:]
        new_game_history.append(move)
        player = self.get_player()
        grid_copy = copy.deepcopy(self.grid)

        bottom_row_index = None
        for row_index in range(NUM_ROWS - 1, -1, -1): # find the bottom of the column
            if self.grid[row_index][move] == 0:
                bottom_row_index = row_index
                break

        # places piece on the grid
        grid_copy[bottom_row_index][move] = player
        outcome = 0
        game_over = False

        # checks if the game has ended in anyway
        # row checker
        for row_index in range(NUM_ROWS):
            for col_index in range(NUM_COLS - 3):
                if grid_copy[row_index][col_index] == player and grid_copy[row_index][col_index+1] == player and grid_copy[row_index][col_index+2] == player and grid_copy[row_index][col_index+3] == player:
                    outcome = player
                    game_over = True

        # column checker
        for row_index in range(NUM_ROWS - 3):
            for col_index in range(NUM_COLS):
                if grid_copy[row_index][col_index] == player and grid_copy[row_index+1][col_index] == player and grid_copy[row_index+2][col_index] == player and grid_copy[row_index+3][col_index] == player:
                    outcome = player
                    game_over = True

        # / checker
        for row_index in range(3, NUM_ROWS):
            for col_index in range(4):
                if grid_copy[row_index][col_index] == player and grid_copy[row_index-1][col_index+1] == player and grid_copy[row_index-2][col_index+2] == player and grid_copy[row_index-3][col_index+3] == player:
                    outcome = player
                    game_over = True

        # \ checker
        for row_index in range(3):
            for col_index in range(4):
                if grid_copy[row_index][col_index] == player and grid_copy[row_index+1][col_index+1] == player and grid_copy[row_index+2][col_index+2] == player and grid_copy[row_index+3][col_index+3] == player:
                    outcome = player
                    game_over = True

        new_game_state = Game_State(
            game_history=new_game_history,
            outcome=outcome,
            game_over=game_over,
        )

        return new_game_state