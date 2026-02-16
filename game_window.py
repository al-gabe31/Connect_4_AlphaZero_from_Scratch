import tkinter as tk
import time
from PIL import Image, ImageTk
from mcts import *

# CONSTANTS
NUM_ROWS = 6
NUM_COLS = 7
CELL_SIZE = 80
PADDING = 10
PIECE_SIZE = 60
WIDTH = NUM_COLS * CELL_SIZE + 2 * PADDING
HEIGHT = NUM_ROWS * CELL_SIZE + 2 * PADDING + 120

BUTTON_WIDTH = 4
BUTTON_HEIGHT = 2
BUTTON_FONT = ('Arial', 14)

class Game_Window:
    def __init__(
        self,
        ai_color:str = None,
        ai_tree:MCTS_Tree = None,
        mcts_simulations:int = None,
        mcts_max_depth:int = None,
        mcts_exploration_constant:int = None
    ):
        # Initialize root window
        self.root = tk.Tk()
        self.root.title('Connect 4')

        # Canvas for board
        self.blue_board = tk.Canvas(self.root, width=WIDTH, height=HEIGHT - 120, bg='blue') # the blue board is where the pieces are placed in
        self.blue_board.pack()

        # Status bar frame (status + buttons)
        # i.e. the widgets you see on the bottom of the screen
        self.status_frame = tk.Frame(self.root, height=60)
        self.status_frame.pack(fill='x')

        # label for the image part of the status (located all the way to the bottom-left side)
        self.status_image_label = tk.Label(self.status_frame)
        self.status_image_label.pack(side='left', padx=10)

        # label for the text part of the status frame (located immediately to the right of status_image_label)
        self.status_text = tk.StringVar()
        self.status_label = tk.Label(self.status_frame, textvariable=self.status_text, font=('Arial', 16))
        self.status_label.pack(side='left', padx=10)

        # Spacer pushes buttons to the right
        # this just adds space between the status and control frame
        self.spacer = tk.Frame(self.status_frame)
        self.spacer.pack(side='left', expand=True, fill='x')

        # Controls frame (history buttons)
        # frame for the control (history) buttons located on the bottom-right side
        self.controls = tk.Frame(self.status_frame)
        self.controls.pack(side='right', padx=10)

        # Load images
        # the images that will be represented for each piece
        self.red_img = Image.open('src/red_piece.png').resize((PIECE_SIZE, PIECE_SIZE), Image.LANCZOS)
        self.yellow_img = Image.open('src/yellow_piece.png').resize((PIECE_SIZE, PIECE_SIZE), Image.LANCZOS)
        self.red_piece = ImageTk.PhotoImage(self.red_img)
        self.yellow_piece = ImageTk.PhotoImage(self.yellow_img)

        # Game state
        self.board = [[None for i in range(NUM_COLS)] for i in range(NUM_ROWS)] # 2D array filled with 'R', 'Y', or None
        self.current_player = 'R' # alternates between 'R' and 'Y'
        self.game_over = False
        self.history = [] # array of integers in [0, 6] that contains the order in which moves are made
        self.history_index = 0 # tells us which turn we're looking at in the history (if history_index == len(history), we're at the current state of the board)
        self.circle_to_column = {}
        self.piece_images = []

        # AI settings
        self.ai_color = ai_color # color of the AI's pieces: 'R', 'Y', or None (None meaning AI isn't playing the game)
        self.ai_tree:MCTS_Tree = ai_tree
        self.mcts_simulations:int = mcts_simulations # how many simulations MCTS will run before getting an output
        self.mcts_max_depth:int = mcts_max_depth # how far at most MCTS will search at the current root node
        self.mcts_exploration_constant:int = mcts_exploration_constant # sets exploration value for MCTS

        # setting history buttons here
        tk.Button(self.controls, text='⏮', command=self.to_start, width=BUTTON_WIDTH, height=BUTTON_HEIGHT, font=BUTTON_FONT).pack(side='left', padx=4)
        tk.Button(self.controls, text='◀', command=self.back_one, width=BUTTON_WIDTH, height=BUTTON_HEIGHT, font=BUTTON_FONT).pack(side='left', padx=4)
        tk.Button(self.controls, text='▶', command=self.forward_one, width=BUTTON_WIDTH, height=BUTTON_HEIGHT, font=BUTTON_FONT).pack(side='left', padx=4)
        tk.Button(self.controls, text='⏭', command=self.to_live, width=BUTTON_WIDTH, height=BUTTON_HEIGHT, font=BUTTON_FONT).pack(side='left', padx=4)

        # binding an on_click events here
        self.blue_board.bind('<Button-1>', self.on_click)

        # if we're playing against an AI, check the game state
        if self.ai_color is not None:
            self.root.after(1, self.check_game_state)

    def reset_board(self):
        self.blue_board.delete('piece') # delete all images labeled 'piece' in the board
        self.piece_images.clear()

        # goes through the entire 2D board and marks each cell as None
        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                self.board[r][c] = None

    def draw_board(self):
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                x1 = PADDING + col * CELL_SIZE + 10
                y1 = PADDING + row * CELL_SIZE + 10
                x2 = x1 + CELL_SIZE - 20
                y2 = y1 + CELL_SIZE - 20
                circle = self.blue_board.create_oval(x1, y1, x2, y2, fill='white', outline='black')
                self.circle_to_column[circle] = col

    def update_status(self, text=None, image=None):
        # if text is provided, change the status message to that text
        if text:
            self.status_text.set(text)

        # if image is provided, change the image in the status section to that image
        if image is not None:
            self.status_image_label.config(image=image)

    def board_full(self):
        # returns true if all pieces on the board are currently occupied
        return all(self.board[0][col] is not None for col in range(NUM_COLS)) # we really only have to check if the "top" cell is occupied for all columns
    
    def check_winner(self, player):
        # something to note about how a winner is checked is that we stop checking the last 3 rows and/or columns
        # this is because there simply wouldn't be enough space left for a connect 4 in that direction (also prevents issues with index out-of-bounds)
        # Horizontal
        for r in range(NUM_ROWS):
            for c in range(NUM_COLS - 3):
                if all(self.board[r][c+i] == player for i in range(4)):
                    return True
                
        # Vertical
        for r in range(NUM_ROWS - 3):
            for c in range(NUM_COLS):
                if all(self.board[r+i][c] == player for i in range(4)):
                    return True
                
        # Diagonal down-right
        for r in range(NUM_ROWS - 3):
            for c in range(NUM_COLS - 3):
                if all(self.board[r+i][c+i] == player for i in range(4)):
                    return True
                
        # Diagonal up-right
        for r in range(3, NUM_ROWS):
            for c in range(NUM_COLS - 3):
                if all(self.board[r-i][c+i] == player for i in range(4)):
                    return True
                
        # if it isn't determined a winner by now, then just return false (no winner found)
        return False
    
    def player_at_index(self, index):
        # when the index is even, it's red's turn
        # otherwise, it's yellow's turn
        return 'R' if index % 2 == 0 else 'Y'
    
    def apply_move(self, col, player):
        for row in reversed(range(NUM_ROWS)): # makes us start checking for an empty spot at the bottom row
            if self.board[row][col] is None: # if the spot is currently unoccupied...
                self.board[row][col] = player # marks an unoccupied cell in the board with a player's piece (annoted as 'R' or 'Y')

                # below deals with setting up the image and placing it exactly on where it should appear on the board
                cx = PADDING + col * CELL_SIZE + CELL_SIZE // 2
                cy = PADDING + row * CELL_SIZE + CELL_SIZE // 2
                img = self.red_piece if player == 'R' else self.yellow_piece
                self.piece_images.append(img)
                self.blue_board.create_image(cx, cy, image=img, tags='piece')
                return
            
        # if all rows in a column are occupied, this function basically does nothing

    def rebuild_from_history(self):
        self.reset_board() # deletes all images on the board and clears piece_images array
        player = 'R'
        for i in range(self.history_index):
            # goes through each turn made up to the history_index and replaces pieces one-by-one as specified in the history array
            self.apply_move(self.history[i], player)
            player = 'Y' if player == 'R' else 'R'

        # Status for history view
        if self.history_index < len(self.history): # i.e. the board isn't the current state of the game
            viewing_player = self.player_at_index(self.history_index) # tells us who's turn it was at that point in history: 'R' or 'Y'
            player_tag = '' if self.ai_color is None else f" [{'AI' if self.ai_color == viewing_player else 'You'}]"
            self.update_status(
                f"{'Red' if viewing_player == 'R' else 'Yellow'} to move{player_tag}",
                self.red_piece if viewing_player == 'R' else self.yellow_piece
            )

    def drop_piece(self, col):
        # can't place a piece if the game is over or if we're currently looking at a past game state (i.e. history_index != len(history))
        if self.game_over or self.history_index != len(self.history):
            return
        
        # otherwise, place the piece on a specified column
        self.apply_move(col, self.current_player)

        # if player is going against AI, let the AI know which move the player chose
        if self.ai_color is not None and self.current_player != self.ai_color:
            self.ai_tree.curr_root_node.full_expansion()
            self.ai_tree.reroot(col)

        # records the new move into the history array and incrementing history_index so that history_index == history.append(col)
        self.history.append(col)
        self.history_index += 1

        # check to see if the current player made a game-winning move
        if self.check_winner(self.current_player):
            self.game_over = True
            self.update_status(
                'Red wins!' if self.current_player == 'R' else 'Yellow wins!',
                self.red_piece if self.current_player == 'R' else self.yellow_piece
            )
            return

        # checks to see if there aren't anymore empty spots on the board
        if self.board_full():
            self.game_over = True
            self.update_status('Tie!', image=None)
            return
        
        self.current_player = 'Y' if self.current_player == 'R' else 'R'
        player_tag = '' if self.ai_color is None else f" [{'AI' if self.ai_color == self.current_player else 'You'}]"
        self.update_status(
            f'Red\'s turn{player_tag}' if self.current_player == 'R' else f'Yellow\'s turn{player_tag}',
            self.red_piece if self.current_player == 'R' else self.yellow_piece
        )

        # check game state if player is going against an AI
        if self.ai_color is not None:
            self.check_game_state()

    # action that will be taken on a click event
    def on_click(self, event):
        # clicks do nothing if we're not looking at the current state of the board or if the game is already over
        if self.history_index != len(self.history) or self.game_over:
            return

        # clicks also do nothing if it's currently the AI's turn
        elif self.current_player == self.ai_color:
            print('It\'s currently the AI\'s turn! Please wait...')
            return

        # otherwise, behave normally
        item = self.blue_board.find_closest(event.x, event.y)
        col = self.circle_to_column.get(item[0])
        if col is not None:
            self.drop_piece(col)

    # handles player vs AI situations
    def check_game_state(self):
        print(f'curr player: {self.current_player}')
        print(f'curr AI history: {"" if len(self.ai_tree.memory_bank) == 0 else self.ai_tree.memory_bank[-1][0]}')
        
        # CASE 1: game is already over
        if self.game_over == True:
            print('Game is over!')

        # CASE 2: AI's turn
        elif self.current_player == self.ai_color:
            print('AI is thinking...')
            
            # get the AI's move
            start_time = time.perf_counter()
            move_chosen = self.ai_tree.make_move(
                max_iterations=self.mcts_simulations,
                max_depth=self.mcts_max_depth,
                exploration_constant=self.mcts_exploration_constant
            )
            end_time = time.perf_counter()
            self.drop_piece(move_chosen)
            print(f'AI chose move {move_chosen} [{round(end_time - start_time, 3)}s]')

        # CASE 3: player's turn
        elif self.current_player != self.ai_color:
            print('Your turn!')

    # history navigation
    def to_start(self):
        if self.ai_color is not None and self.ai_color == self.current_player and self.game_over == False:
            # can't use history button if it's currently the AI's turn
            print('currently AI\'s turn!')
            return
        
        self.history_index = 0
        self.rebuild_from_history() # for this case, it will just clear the board
        self.update_status('Viewing start of game', image=None)

    def back_one(self):
        if self.ai_color is not None and self.ai_color == self.current_player and self.game_over == False:
            # can't use history button if it's currently the AI's turn
            print('currently AI\'s turn!')
            return
        
        if self.history_index > 0: # prevents history_index being negative
            self.history_index -= 1
            self.rebuild_from_history() # clears the board and replaces all pieces up to the last turn made
            curr_player = 'R' if self.history_index % 2 == 0 else 'Y'
            player_tag = '' if self.ai_color is None else f" [{'AI' if self.ai_color == curr_player else 'You'}]"
            self.update_status(f'Move {self.history_index}/{len(self.history)}{player_tag}')

    def forward_one(self):
        if self.ai_color is not None and self.ai_color == self.current_player and self.game_over == False:
            # can't use history button if it's currently the AI's turn
            print('currently AI\'s turn!')
            return
        
        if self.history_index < len(self.history): # prevents history_index going outside history
            self.history_index += 1
            self.rebuild_from_history() # clears the board and replaces all pieces up to the next turns in history view

            # check to see if the last turn of the game is either a game-winning move, a tie, or if the game ist still ongoing
            if self.check_winner('R' if self.history_index % 2 == 1 else 'Y'): # the last move made was game-winning
                self.update_status(
                    'Red wins!' if self.history_index % 2 == 1 else 'Yellow wins!',
                    self.red_piece if self.history_index % 2 == 1 else self.yellow_piece
                )
            elif self.board_full():
                # last move wasn't game winning...
                # and the board is full means it's a tie
                self.update_status('Tie!', image=None)
            else:
                curr_player = 'R' if self.history_index % 2 == 0 else 'Y'
                player_tag = '' if self.ai_color is None else f" [{'AI' if self.ai_color == curr_player else 'You'}]"
                self.update_status(
                    f'Move {self.history_index}/{len(self.history)}{player_tag}',
                    self.red_piece if self.history_index % 2 == 0 else self.yellow_piece
                )

    def to_live(self):
        if self.ai_color is not None and self.ai_color == self.current_player and self.game_over == False:
            # can't use history button if it's currently the AI's turn
            print('currently AI\'s turn!')
            return
        
        self.history_index = len(self.history) # make history_index catch up to the current state of the board
        self.rebuild_from_history() # clears the board and replaces all the pieces up to the current state of the board

        # check to see if the last turn of the game is either a game-winning move, a tie or if the game is still ongoing
        if self.check_winner('R' if self.history_index % 2 == 1 else 'Y'): # the last move made was game-winning
            self.update_status(
                'Red wins!' if self.history_index % 2 == 1 else 'Yellow wins!',
                self.red_piece if self.history_index % 2 == 1 else self.yellow_piece
            )
        elif self.board_full():
            # last move made wasn't game-winning...
            # and the board is full means it's a tie
            self.update_status('Tie!', image=None)
        else:
            player_tag = '' if self.ai_color is None else f" [{'AI' if self.ai_color == self.current_player else 'You'}]"
            self.update_status(
                f'Red\'s turn{player_tag}' if self.history_index % 2 == 0 else f'Yellow\'s turn{player_tag}',
                self.red_piece if self.history_index % 2 == 0 else self.yellow_piece
            )

def run_2_player_game():
    window = Game_Window()
    window.draw_board()
    window.update_status('Red\'s turn', window.red_piece)
    window.root.mainloop()

def load_history(imported_history):
    window = Game_Window()
    window.draw_board()

    window.history = []
    window.history_index = 0
    window.current_player = 'R'
    window.game_over = False
    window.reset_board()

    player = 'R'
    for move in imported_history: # move = column
        # if the move is outside the range [0, 6], raise an error for invalid move in imported_history
        if move < 0 or move >= NUM_COLS:
            raise ValueError(f'Invalid column: {move}')
        
        # find the next available spot in the column to place the next piece
        for row in reversed(range(NUM_ROWS)):
            if window.board[row][move] is None:
                window.board[row][move] = player # marks a spot on the 2D grid with a player
                window.history.append(move) # add to history
                window.history_index += 1 # increment turn #

                # below deals with setting up the piece image and placing it on the board
                cx = PADDING + move * CELL_SIZE + CELL_SIZE // 2
                cy = PADDING + row * CELL_SIZE + CELL_SIZE // 2
                img = window.red_piece if player == 'R' else window.yellow_piece
                window.piece_images.append(img)
                window.blue_board.create_image(cx, cy, image=img, tags='piece')

                # check to see if the player made a game-winning move
                if window.check_winner(player):
                    window.game_over = True
                    window.update_status(
                        'Red wins!' if player == 'R' else 'Yellow wins!',
                        img
                    )
                    return
                
                # if the board is full, the game is over
                if window.board_full():
                    window.game_over = True
                    window.update_status('Tie!', image=None)
                    return
                player = 'Y' if player == 'R' else 'R'
                break
        else:
            raise ValueError(f'Column {move + 1} is full')
            
    window.history_index = len(window.history)
    window.current_player = player
    window.update_status(
        'Red\'s turn' if window.current_player == 'R' else 'Yellow\'s turn',
        window.red_piece if window.current_player == 'R' else window.yellow_piece
    )

    window.root.mainloop()

def play_against_AI(
        player_going_first:bool,
        neural_network:Neural_Network,
        mcts_simulations:int = 100,
        mcts_max_depth:int = 6,
        mcts_exploration_constant:int = 3
):
    # ==================== GAME SETUP ==================== #
    player_color = 'R' if player_going_first else 'Y' # player is red if they're going first, otherwise yellow
    ai_color = 'R' if player_color == 'Y' else 'Y' # ai gets the other color

    print(f'player_color = {player_color}')
    print(f'ai_color = {ai_color}')

    # setting up the MCTS tree for the AI
    tree = MCTS_Tree(
        root_history=[],
        neural_network=neural_network
    )

    # setting up the window for the game to run in
    window = Game_Window(
        ai_color=ai_color,
        ai_tree=tree,
        mcts_simulations=mcts_simulations,
        mcts_max_depth=mcts_max_depth,
        mcts_exploration_constant=mcts_exploration_constant
    )
    window.draw_board()
    window.update_status('Red\'s turn', window.red_piece)
    window.root.mainloop()



if __name__ == "__main__":
    history_1 = [6, 2, 3, 3, 6, 6, 3, 2, 3, 2, 2, 3, 5, 2]
    history_2 = [2, 6, 3, 3, 6, 2, 0, 6]

    load_history(history_1)
    load_history(history_2)
    
    print('all good!')