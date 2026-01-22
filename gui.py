import tkinter as tk
from PIL import Image, ImageTk

ROWS = 6
COLS = 7
CELL_SIZE = 80
PADDING = 10
PIECE_SIZE = 60

WIDTH = COLS * CELL_SIZE + 2 * PADDING
HEIGHT = ROWS * CELL_SIZE + 2 * PADDING + 120

root = tk.Tk()
root.title("Connect 4")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT - 120, bg="blue")
canvas.pack()

# --- Status bar ---
status_frame = tk.Frame(root, height=60)
status_frame.pack(fill="x")

status_image_label = tk.Label(status_frame)
status_image_label.pack(side="left", padx=10)

status_text = tk.StringVar()
status_label = tk.Label(status_frame, textvariable=status_text, font=("Arial", 16))
status_label.pack(side="left")

# --- Controls ---
controls = tk.Frame(root, height=60)
controls.pack(fill="x")

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

# --- Draw board ---
def draw_board():
    for row in range(ROWS):
        for col in range(COLS):
            x1 = PADDING + col * CELL_SIZE + 10
            y1 = PADDING + row * CELL_SIZE + 10
            x2 = x1 + CELL_SIZE - 20
            y2 = y1 + CELL_SIZE - 20

            circle = canvas.create_oval(
                x1, y1, x2, y2,
                fill="white",
                outline="black"
            )
            circle_to_column[circle] = col

draw_board()

def update_status(text=None, image=None):
    if text:
        status_text.set(text)
    if image:
        status_image_label.config(image=image)

def reset_board():
    canvas.delete("piece")
    piece_images.clear()
    for r in range(ROWS):
        for c in range(COLS):
            board[r][c] = None

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

def check_winner(player):
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c+i] == player for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r+i][c] == player for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r+i][c+i] == player for i in range(4)):
                return True
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r-i][c+i] == player for i in range(4)):
                return True
    return False

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

    current_player = "Y" if current_player == "R" else "R"
    update_status(
        "Red's turn" if current_player == "R" else "Yellow's turn",
        red_piece if current_player == "R" else yellow_piece
    )

def on_click(event):
    if history_index != len(history):
        return
    item = canvas.find_closest(event.x, event.y)
    col = circle_to_column.get(item[0])
    if col is not None:
        drop_piece(col)

canvas.bind("<Button-1>", on_click)

# --- History controls ---
def to_start():
    global history_index
    history_index = 0
    rebuild_from_history()
    update_status("Viewing start of game")

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

tk.Button(controls, text="⏮", command=to_start, width=5).pack(side="left", padx=5)
tk.Button(controls, text="◀", command=back_one, width=5).pack(side="left")
tk.Button(controls, text="▶", command=forward_one, width=5).pack(side="left")
tk.Button(controls, text="⏭", command=to_live, width=5).pack(side="left", padx=5)

update_status("Red's turn", red_piece)
root.mainloop()
