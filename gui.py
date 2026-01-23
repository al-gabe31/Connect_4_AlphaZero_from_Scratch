import tkinter as tk
from PIL import Image, ImageTk

# --- Constants ---
NUM_ROWS = 6
NUM_COLS = 7
CELL_SIZE = 80
PADDING = 10
PIECE_SIZE = 60
WIDTH = NUM_COLS * CELL_SIZE + 2 * PADDING
HEIGHT = NUM_ROWS * CELL_SIZE + 2 * PADDING + 120

# --- Initialize root window ---
root = tk.Tk()
root.title("Connect 4")

# --- Canvas for board ---
blue_board = tk.Canvas(root, width=WIDTH, height=HEIGHT - 120, bg="blue") # the blue board the pieces are placed in
blue_board.pack()

# --- Status bar frame (status + buttons) ---
# i.e. the widgets you see on the bottom of the screen
status_frame = tk.Frame(root, height=60)
status_frame.pack(fill="x")

# label for the image part of the status (located all the way to the bottom-left side)
status_image_label = tk.Label(status_frame)
status_image_label.pack(side="left", padx=10)

# label for the text part of the status frame (located immediately to the right of status_image_label)
status_text = tk.StringVar()
status_label = tk.Label(status_frame, textvariable=status_text, font=("Arial", 16))
status_label.pack(side="left", padx=10)

# Spacer pushes buttons to the right
# this just adds space between the status and control frame
spacer = tk.Frame(status_frame)
spacer.pack(side="left", expand=True, fill="x")

# --- Controls frame (history buttons) ---
# frame for the control (history) buttons located on the bottom-right side
controls = tk.Frame(status_frame)
controls.pack(side="right", padx=10)

# --- Load images ---
# the images that will be represented for each piece
red_img = Image.open("src/red_piece.png").resize((PIECE_SIZE, PIECE_SIZE), Image.LANCZOS)
yellow_img = Image.open("src/yellow_piece.png").resize((PIECE_SIZE, PIECE_SIZE), Image.LANCZOS)
red_piece = ImageTk.PhotoImage(red_img)
yellow_piece = ImageTk.PhotoImage(yellow_img)

# --- Game state ---
board = [[None for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)] # 2D array filled with 'R', 'Y', or None
current_player = "R" # alternates between 'R' and 'Y'
game_over = False
history = [] # array of integers in [0, 6] that contains the order in which moves are made
history_index = 0 # tells us which turn we're looking at in the history (if history_index == len(hostory), we're at the current state of the board)
circle_to_column = {}
piece_images = []

# --- Helper functions ---
def reset_board():
    blue_board.delete("piece") # delete all images labeled 'piece' in the board
    piece_images.clear()

    # goes through the entire 2D board and marks each cell as None
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            board[r][c] = None

def draw_board():
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            x1 = PADDING + col * CELL_SIZE + 10
            y1 = PADDING + row * CELL_SIZE + 10
            x2 = x1 + CELL_SIZE - 20
            y2 = y1 + CELL_SIZE - 20
            circle = blue_board.create_oval(x1, y1, x2, y2, fill="white", outline="black")
            circle_to_column[circle] = col

def update_status(text=None, image=None):

    # if text is provided, change the status message to that text
    if text:
        status_text.set(text)

    # if image is provided, change the image in the status section to that image
    if image is not None:
        status_image_label.config(image=image)

def board_full():
    # returns true if all pieces on the board are currently occupied
    return all(board[0][col] is not None for col in range(NUM_COLS))

def check_winner(player):

    # something to note about how a winner is checked is that we stop checking the last 3 rows and/or columns
    # this is because there simply wouldn't be enough space left for a connect 4 in that direction (also prevents issues with index out-of-bounds)
    # Horizontal
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS - 3):
            if all(board[r][c+i] == player for i in range(4)):
                return True
    # Vertical
    for r in range(NUM_ROWS - 3):
        for c in range(NUM_COLS):
            if all(board[r+i][c] == player for i in range(4)):
                return True
    # Diagonal down-right
    for r in range(NUM_ROWS - 3):
        for c in range(NUM_COLS - 3):
            if all(board[r+i][c+i] == player for i in range(4)):
                return True
    # Diagonal up-right
    for r in range(3, NUM_ROWS):
        for c in range(NUM_COLS - 3):
            if all(board[r-i][c+i] == player for i in range(4)):
                return True
            
    # if it isn't determined a winner by now, then just return false (no winner found)
    return False

def player_at_index(index):
    # when the inde is even, it's red's turn
    # otherwise, it's yellow's turn
    return "R" if index % 2 == 0 else "Y"

def apply_move(col, player):
    for row in reversed(range(NUM_ROWS)): # makes us start checking for an empty spot at the bottom row
        if board[row][col] is None: # if the spot is currently unoccupied...
            board[row][col] = player # marks an unoccupied cell in the board with a player's piece (annoted as 'R' or 'Y')
            
            # below deals with setting up the image and placing it exactly on where it should appear on the board
            cx = PADDING + col * CELL_SIZE + CELL_SIZE // 2
            cy = PADDING + row * CELL_SIZE + CELL_SIZE // 2
            img = red_piece if player == "R" else yellow_piece
            piece_images.append(img)
            blue_board.create_image(cx, cy, image=img, tags="piece")
            return
        
    # if all rows in a column are occupied, this funciton basically does nothing

def rebuild_from_history():
    reset_board() # deletes all images on the board and clears piece_images array
    player = "R"
    for i in range(history_index):
        # goes through each turn made up to the history_index and replaces pieces one-by-one as specified in the history array
        apply_move(history[i], player)
        player = "Y" if player == "R" else "R"

    # Status for history view
    if history_index < len(history): # i.e. the board isn't the current state of the game
        viewing_player = player_at_index(history_index) # tells us who's turn it was at that point in history: 'R' or 'Y'
        update_status(
            f"{'Red' if viewing_player == 'R' else 'Yellow'} to move",
            red_piece if viewing_player == "R" else yellow_piece
        )

def drop_piece(col):
    global current_player, game_over, history_index

    # can't place a piece if the game is over or if we're currently looking at a past game state (i.e. history_index != len(history))
    if game_over or history_index != len(history):
        return

    # otherwise, place the piece on a specified column
    apply_move(col, current_player)

    # records the new move into the history array and incrementing history_index so that history_index == history.append(col)
    history.append(col)
    history_index += 1

    # check to see if the current player made a game-winning move
    if check_winner(current_player):
        game_over = True
        update_status(
            "Red wins!" if current_player == "R" else "Yellow wins!",
            red_piece if current_player == "R" else yellow_piece
        )
        return

    # checks to see if there aren't any more empty spots on the board
    if board_full():
        game_over = True
        update_status("Tie!", image=None)
        return

    current_player = "Y" if current_player == "R" else "R"
    update_status(
        "Red's turn" if current_player == "R" else "Yellow's turn",
        red_piece if current_player == "R" else yellow_piece
    )

# action that will be taken on a click event
def on_click(event):

    # clicks do nothing if we're not looking at the current state of the board or if the game is already over
    if history_index != len(history) or game_over:
        return
    item = blue_board.find_closest(event.x, event.y)
    col = circle_to_column.get(item[0])
    if col is not None:
        drop_piece(col)

blue_board.bind("<Button-1>", on_click)

# --- History navigation ---
def to_start():
    global history_index
    history_index = 0
    rebuild_from_history() # for this case, it will just clear the board
    update_status("Viewing start of game", image=None)

def back_one():
    global history_index
    if history_index > 0: # prevents history_index being negative
        history_index -= 1
        rebuild_from_history() # clears the board and replaces all pieces up to the last turn made
        update_status(f"Move {history_index}/{len(history)}")

def forward_one():
    global history_index
    if history_index < len(history): # prevents history_index going outside history
        history_index += 1
        rebuild_from_history() # clears the board and replaces all pieces up to the next turn in history view

        # check to see if the last turn of the game is either a game-wining move, a tie, or if the game is still ongoing
        if check_winner('R' if history_index % 2 == 1 else 'Y'): # the last move made was game-winning
            update_status(
                'Red wins!' if history_index % 2 == 1 else 'Yellow wins!',
                red_piece if history_index % 2 == 1 else yellow_piece
            )
        elif board_full():
            # last move made wasn't game winning...
            # and the board is full means it's a tie
            update_status('Tie!', image=None)
        else:
            update_status(
                f"Move {history_index}/{len(history)}",
                red_piece if history_index % 2 == 0 else yellow_piece
            )

def to_live():
    global history_index
    history_index = len(history) # make history_index catch up to the current state of the board
    rebuild_from_history() # clears the board and replaces all the pieces up to the current state of the board

    # check to see if the last turn of the game is either a game-winning move, a tie, or if the game is still ongoing
    if check_winner('R' if history_index % 2 == 1 else 'Y'): # the last move made was game-winning
        update_status(
            'Red wins!' if history_index % 2 == 1 else 'Yellow wins!',
            red_piece if history_index % 2 == 1 else yellow_piece
        )
    elif board_full():
        # last move made wasn't game-winning...
        # and the board is full means it's a tie
        update_status('Tie!', image=None)
    else:
        update_status(
            "Red's turn" if history_index % 2 == 0 else "Yellow's turn",
            red_piece if history_index % 2 == 0 else yellow_piece
        )

# --- Buttons ---
# customize the history buttons here!
BUTTON_WIDTH = 4
BUTTON_HEIGHT = 2
BUTTON_FONT = ("Arial", 14)

tk.Button(controls, text="⏮", command=to_start,
          width=BUTTON_WIDTH, height=BUTTON_HEIGHT, font=BUTTON_FONT).pack(side="left", padx=4)
tk.Button(controls, text="◀", command=back_one,
          width=BUTTON_WIDTH, height=BUTTON_HEIGHT, font=BUTTON_FONT).pack(side="left", padx=4)
tk.Button(controls, text="▶", command=forward_one,
          width=BUTTON_WIDTH, height=BUTTON_HEIGHT, font=BUTTON_FONT).pack(side="left", padx=4)
tk.Button(controls, text="⏭", command=to_live,
          width=BUTTON_WIDTH, height=BUTTON_HEIGHT, font=BUTTON_FONT).pack(side="left", padx=4)

# --- Load a game history ---
def load_history(imported_history):
    global history, history_index, current_player, game_over

    history = []
    history_index = 0
    current_player = "R"
    game_over = False
    reset_board()

    player = "R"
    for move in imported_history: # move = column

        # if the move is outside the range [0, 6], raise an error for invalid move in imported_history
        if move < 0 or move >= NUM_COLS:
            raise ValueError(f"Invalid column: {move}")
        
        # find the next available spot in the column to place the next piece
        for row in reversed(range(NUM_ROWS)):
            if board[row][move] is None:
                board[row][move] = player # marks a spot on the 2D grid with a player
                history.append(move) # add to history
                history_index += 1 # increment turn #

                # below deals with setting up the piece image and placing it on the board
                cx = PADDING + move * CELL_SIZE + CELL_SIZE // 2
                cy = PADDING + row * CELL_SIZE + CELL_SIZE // 2
                img = red_piece if player == "R" else yellow_piece
                piece_images.append(img)
                blue_board.create_image(cx, cy, image=img, tags="piece")

                # check to see if the player made a game-winning move
                if check_winner(player):
                    game_over = True
                    update_status(
                        "Red wins!" if player == "R" else "Yellow wins!",
                        img
                    )
                    return
                
                # if the board is full, the game is over
                if board_full():
                    game_over = True
                    update_status("Tie!", image=None)
                    return
                player = "Y" if player == "R" else "R"
                break
        else:
            raise ValueError(f"Column {move + 1} is full")

    history_index = len(history)
    current_player = player
    update_status(
        "Red's turn" if current_player == "R" else "Yellow's turn",
        red_piece if current_player == "R" else yellow_piece
    )

# --- Draw initial board ---
draw_board()
update_status("Red's turn", red_piece)

# --- Start main loop only if run as script ---
if __name__ == "__main__":
    root.mainloop()
