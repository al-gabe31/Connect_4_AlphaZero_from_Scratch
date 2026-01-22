import tkinter as tk
from PIL import Image, ImageTk

# --- Constants ---
ROWS = 6
COLS = 7
CELL_SIZE = 80
PADDING = 10
PIECE_SIZE = 60
WIDTH = COLS * CELL_SIZE + 2 * PADDING
HEIGHT = ROWS * CELL_SIZE + 2 * PADDING + 120

# --- Initialize root window ---
root = tk.Tk()
root.title("Connect 4")

# --- Canvas for board ---
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT - 120, bg="blue")
canvas.pack()

# --- Status bar frame (status + buttons) ---
status_frame = tk.Frame(root, height=60)
status_frame.pack(fill="x")

status_image_label = tk.Label(status_frame)
status_image_label.pack(side="left", padx=10)

status_text = tk.StringVar()
status_label = tk.Label(status_frame, textvariable=status_text, font=("Arial", 16))
status_label.pack(side="left", padx=10)

# Spacer pushes buttons to the right
spacer = tk.Frame(status_frame)
spacer.pack(side="left", expand=True, fill="x")

# --- Controls frame (history buttons) ---
controls = tk.Frame(status_frame)
controls.pack(side="right", padx=10)

# --- Load images ---
red_img = Image.open("src/red_piece.png").resize((PIECE_SIZE, PIECE_SIZE), Image.LANCZOS)
yellow_img = Image.open("src/yellow_piece.png").resize((PIECE_SIZE, PIECE_SIZE), Image.LANCZOS)
red_piece = ImageTk.PhotoImage(red_img)
yellow_piece = ImageTk.PhotoImage(yellow_img)

# --- Game state ---
board = [[None for _ in range(COLS)] for _ in range(ROWS)]
current_player = "R"
game_over = False
history = []
history_index = 0
circle_to_column = {}
piece_images = []

# --- Helper functions ---
def reset_board():
    canvas.delete("piece")
    piece_images.clear()
    for r in range(ROWS):
        for c in range(COLS):
            board[r][c] = None

def draw_board():
    for row in range(ROWS):
        for col in range(COLS):
            x1 = PADDING + col * CELL_SIZE + 10
            y1 = PADDING + row * CELL_SIZE + 10
            x2 = x1 + CELL_SIZE - 20
            y2 = y1 + CELL_SIZE - 20
            circle = canvas.create_oval(x1, y1, x2, y2, fill="white", outline="black")
            circle_to_column[circle] = col

def update_status(text=None, image=None):
    if text:
        status_text.set(text)
    if image is not None:
        status_image_label.config(image=image)

def board_full():
    return all(board[0][col] is not None for col in range(COLS))

def check_winner(player):
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c+i] == player for i in range(4)):
                return True
    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r+i][c] == player for i in range(4)):
                return True
    # Diagonal down-right
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r+i][c+i] == player for i in range(4)):
                return True
    # Diagonal up-right
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r-i][c+i] == player for i in range(4)):
                return True
    return False

def player_at_index(index):
    return "R" if index % 2 == 0 else "Y"

def apply_move(col, player):
    for row in reversed(range(ROWS)):
        if board[row][col] is None:
            board[row][col] = player
            cx = PADDING + col * CELL_SIZE + CELL_SIZE // 2
            cy = PADDING + row * CELL_SIZE + CELL_SIZE // 2
            img = red_piece if player == "R" else yellow_piece
            piece_images.append(img)
            canvas.create_image(cx, cy, image=img, tags="piece")
            return

def rebuild_from_history():
    reset_board()
    player = "R"
    for i in range(history_index):
        apply_move(history[i], player)
        player = "Y" if player == "R" else "R"

    # Status for history view
    if history_index < len(history):
        viewing_player = player_at_index(history_index)
        update_status(
            f"{'Red' if viewing_player == 'R' else 'Yellow'} to move",
            red_piece if viewing_player == "R" else yellow_piece
        )

def drop_piece(col):
    global current_player, game_over, history_index

    if game_over or history_index != len(history):
        return

    apply_move(col, current_player)
    history.append(col)
    history_index += 1

    if check_winner(current_player):
        game_over = True
        update_status(
            "Red wins!" if current_player == "R" else "Yellow wins!",
            red_piece if current_player == "R" else yellow_piece
        )
        return

    if board_full():
        game_over = True
        update_status("Tie!", image=None)
        return

    current_player = "Y" if current_player == "R" else "R"
    update_status(
        "Red's turn" if current_player == "R" else "Yellow's turn",
        red_piece if current_player == "R" else yellow_piece
    )

def on_click(event):
    if history_index != len(history) or game_over:
        return
    item = canvas.find_closest(event.x, event.y)
    col = circle_to_column.get(item[0])
    if col is not None:
        drop_piece(col)

canvas.bind("<Button-1>", on_click)

# --- History navigation ---
def to_start():
    global history_index
    history_index = 0
    rebuild_from_history()
    update_status("Viewing start of game", image=None)

def back_one():
    global history_index
    if history_index > 0:
        history_index -= 1
        rebuild_from_history()
        update_status(f"Move {history_index}/{len(history)}")

def forward_one():
    global history_index
    if history_index < len(history):
        history_index += 1
        rebuild_from_history()
        update_status(f"Move {history_index}/{len(history)}")

def to_live():
    global history_index
    history_index = len(history)
    rebuild_from_history()
    update_status(
        "Red's turn" if current_player == "R" else "Yellow's turn",
        red_piece if current_player == "R" else yellow_piece
    )

# --- Buttons ---
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
def load_history(new_history):
    global history, history_index, current_player, game_over

    history = []
    history_index = 0
    current_player = "R"
    game_over = False
    reset_board()

    player = "R"
    for move in new_history:
        if move < 0 or move >= COLS:
            raise ValueError(f"Invalid column: {move}")
        for row in reversed(range(ROWS)):
            if board[row][move] is None:
                board[row][move] = player
                history.append(move)
                history_index += 1
                cx = PADDING + move * CELL_SIZE + CELL_SIZE // 2
                cy = PADDING + row * CELL_SIZE + CELL_SIZE // 2
                img = red_piece if player == "R" else yellow_piece
                piece_images.append(img)
                canvas.create_image(cx, cy, image=img, tags="piece")
                if check_winner(player):
                    game_over = True
                    update_status(
                        "Red wins!" if player == "R" else "Yellow wins!",
                        img
                    )
                    return
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
