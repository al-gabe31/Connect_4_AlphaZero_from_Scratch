import tkinter as tk
from PIL import Image, ImageTk

ROWS = 6
COLS = 7
CELL_SIZE = 80
PADDING = 10

WIDTH = COLS * CELL_SIZE + 2 * PADDING
HEIGHT = ROWS * CELL_SIZE + 2 * PADDING

root = tk.Tk()
root.title("Connect 4")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="blue")
canvas.pack()

# --- Load images (keep references!) ---
PIECE_SIZE = 60  # try 56–64 if you want to tweak

red_img = Image.open("src/red_piece.png").resize((PIECE_SIZE, PIECE_SIZE), Image.LANCZOS)
yellow_img = Image.open("src/yellow_piece.png").resize((PIECE_SIZE, PIECE_SIZE), Image.LANCZOS)

red_piece = ImageTk.PhotoImage(red_img)
yellow_piece = ImageTk.PhotoImage(yellow_img)


# --- Game state ---
board = [[None for _ in range(COLS)] for _ in range(ROWS)]
current_player = "R"

# Map canvas items to columns
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

def drop_piece(col):
    global current_player

    # Find lowest empty row
    for row in reversed(range(ROWS)):
        if board[row][col] is None:
            board[row][col] = current_player

            cx = PADDING + col * CELL_SIZE + CELL_SIZE // 2
            cy = PADDING + row * CELL_SIZE + CELL_SIZE // 2

            img = red_piece if current_player == "R" else yellow_piece
            canvas.create_image(cx, cy, image=img)

            current_player = "Y" if current_player == "R" else "R"
            return

    print(f"Column {col + 1} is full!")

def on_click(event):
    item = canvas.find_closest(event.x, event.y)
    if not item:
        return

    col = circle_to_column.get(item[0])
    if col is not None:
        print(f"Clicked column: {col + 1}")
        drop_piece(col)

canvas.bind("<Button-1>", on_click)

root.mainloop()
