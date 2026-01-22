import tkinter as tk
from PIL import Image, ImageTk

ROWS = 6
COLS = 7
CELL_SIZE = 80
PADDING = 10
PIECE_SIZE = 60

WIDTH = COLS * CELL_SIZE + 2 * PADDING
HEIGHT = ROWS * CELL_SIZE + 2 * PADDING + 80  # space for status bar

root = tk.Tk()
root.title("Connect 4")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT - 80, bg="blue")
canvas.pack()

# --- Status bar ---
status_frame = tk.Frame(root, height=80)
status_frame.pack(fill="x")

status_image_label = tk.Label(status_frame)
status_image_label.pack(side="left", padx=20)

status_text = tk.StringVar()
status_label = tk.Label(status_frame, textvariable=status_text, font=("Arial", 16))
status_label.pack(side="left")

# --- Load images ---
red_img = Image.open("src/red_piece.png").resize((PIECE_SIZE, PIECE_SIZE), Image.LANCZOS)
yellow_img = Image.open("src/yellow_piece.png").resize((PIECE_SIZE, PIECE_SIZE), Image.LANCZOS)

red_piece = ImageTk.PhotoImage(red_img)
yellow_piece = ImageTk.PhotoImage(yellow_img)

# --- Game state ---
board = [[None for _ in range(COLS)] for _ in range(ROWS)]
current_player = "R"
game_over = False

circle_to_column = {}

# --- Draw board ---
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

def update_status():
    if current_player == "R":
        status_image_label.config(image=red_piece)
        status_text.set("Red's turn")
    else:
        status_image_label.config(image=yellow_piece)
        status_text.set("Yellow's turn")

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

def drop_piece(col):
    global current_player, game_over

    if game_over:
        return

    for row in reversed(range(ROWS)):
        if board[row][col] is None:
            board[row][col] = current_player

            cx = PADDING + col * CELL_SIZE + CELL_SIZE // 2
            cy = PADDING + row * CELL_SIZE + CELL_SIZE // 2

            img = red_piece if current_player == "R" else yellow_piece
            canvas.create_image(cx, cy, image=img)

            if check_winner(current_player):
                game_over = True
                status_image_label.config(image=img)
                status_text.set(
                    "Red wins!" if current_player == "R" else "Yellow wins!"
                )
                return

            current_player = "Y" if current_player == "R" else "R"
            update_status()
            return

def on_click(event):
    if game_over:
        return

    item = canvas.find_closest(event.x, event.y)
    col = circle_to_column.get(item[0])
    if col is not None:
        drop_piece(col)

canvas.bind("<Button-1>", on_click)

update_status()
root.mainloop()
