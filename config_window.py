import tkinter as tk

# class for round buttons
class Selectable_Round_Button(tk.Canvas):
    def __init__(
            self,
            parent,
            text,
            group=None,
            command=None,
            width=140,
            height=40,
            corner_radius=12,
            border_width=4,
            border_color='black',
            fill_color='#d9d9d9',
            hover_color='#c9c9c9',
            text_color='black',
            font=('Arial', 10, 'bold'),
            default_selected=False
    ):
        # ==================== BUTTON SETUP ==================== #
        super().__init__(
            parent,
            width=width,
            height=height,
            highlightthickness=0,
            bg=parent['bg']
        )

        self.command = command
        self.group = group
        self.fill_color = fill_color
        self.hover_color = hover_color
        self.selected = False

        # creating the button itself
        self.rect = self.create_round_rect(
            border_width,
            border_width,
            width - border_width,
            height - border_width,
            corner_radius,
            outline=border_color,
            width=border_width,
            fill=fill_color
        )

        self.text = self.create_text(
            width/2,
            height/2,
            text=text,
            fill=text_color,
            font=font
        )



        # ==================== BUTTON FUNCTIONALITY ==================== #
        self.bind('<Button-1>', self.on_click)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

        # add to group if provided
        if group is not None:
            group.append(self)

        # default selection
        if default_selected:
            self.select()

    def create_round_rect(
            self,
            x1,
            y1,
            x2,
            y2,
            r=25,
            **kwargs
    ):
        points = [
            x1+r, y1,
            x2-r, y1,
            x2, y1,
            x2, y1+r,
            x2, y2-r,
            x2, y2,
            x2-r, y2,
            x1+r, y2,
            x1, y2,
            x1, y2-r,
            x1, y1+r,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    # handles event where the player clicks on a button
    def select(self):
        if self.group:
            for btn in self.group:
                btn.deselect()

        self.selected = True
        self.itemconfig(self.rect, fill=self.hover_color)

        if self.command:
            self.command()

    # handles event where a button is deselected
    def deselect(self):
        self.selected = False
        self.itemconfig(self.rect, fill=self.fill_color)

    # effects of when the player clicks on a button
    def on_click(self, event):
        self.select()

    # effects of when the player hovers on a button
    def on_enter(self, event):
        if not self.selected:
            self.itemconfig(self.rect, fill=self.hover_color)

    # effects of when the player unhovers a button
    def on_leave(self, event):
        if not self.selected:
            self.itemconfig(self.rect, fill=self.fill_color)



# class for the initial config window
class Init_Config_Window:
    def __init__(self):
        # ==================== EXTRACTABLE SETTINGS ==================== #
        self.opponent_is_player = True # True for player vs player, False for player vs AI



        # ==================== CREATING MAIN WINDOW ==================== #
        # window setup
        self.root = tk.Tk()
        self.root.title('Connect 4 Initial Setup')
        self.root.configure(bg='#f0f0f0')

        # text & button layout setup
        self.create_header(self.root, 'CHOOSE OPPONENT')

        self.opponent_frame = tk.Frame(self.root, bg=self.root['bg'])
        self.opponent_frame.pack(pady=5)

        # button for player opponent
        Selectable_Round_Button(
            self.opponent_frame,
            'PLAYER',
            width=200,
            height=50,
            group=None,
            command=lambda: self.opponent_chosen(self.root, 'Player'),
            border_color='#00c452',
            fill_color='#76ffaf',
            hover_color='#00eb62'
        ).pack(side='left', padx=6)

        # button for AI opponent
        Selectable_Round_Button(
            self.opponent_frame,
            'AI',
            width=200,
            height=50,
            group=None,
            command=lambda: self.opponent_chosen(self.root, 'AI'),
            border_color='#e60000',
            fill_color='#ffb4b4',
            hover_color='#ff8181'
        ).pack(side='left', padx=6)

    def opponent_chosen(self, root, chosen_opponent):
        # updates opponent setting=
        if chosen_opponent == 'Player':
            self.opponent_is_player = True
        elif chosen_opponent == 'AI':
            self.opponent_is_player = False

        root.destroy() # closes the window

    def create_header(self, parent, text):
        label = tk.Label(
            parent,
            text=text,
            font=('Arial', 12, 'bold'),
            bg=parent['bg']
        )
        label.pack(pady=(8, 2))



# class for the AI config window
class AI_Config_Window:
    def __init__(self):
        # ==================== EXTRACTABLE SETTINGS ==================== #
        self.player_color = 'R' # defaults player playing red (1st)
        self.ai_settings = { # defaults easy setting
            'mcts_simulations': 100,
            'mcts_max_depth': 6,
            'mcts_exploration_constant': 3
        }



        # ==================== CREATING MAIN WINDOW ==================== #
        # window setup
        self.root = tk.Tk()
        self.root.title('Connect 4 AI Setup')
        self.root.configure(bg='#f0f0f0')

        # group for exclusive selection
        self.color_group = []
        self.difficulty_group = []



        # ==================== ROW 1 - PLAYER COLOR ==================== #
        self.create_header(self.root, 'PLAYER COLOR')

        self.color_frame = tk.Frame(self.root, bg=self.root['bg'])
        self.color_frame.pack(pady=5)

        # button for the player picking red
        Selectable_Round_Button(
            self.color_frame, 
            'RED [1st]',
            group=self.color_group,
            command=lambda: self.select_color('Red'),
            width=215,
            default_selected=True,
            border_color='#e60000',
            fill_color='#ffffff',
            hover_color='#ff8181'
        ).pack(side='left', padx=8)

        # button for the player picking yellow
        Selectable_Round_Button(
            self.color_frame, 
            'YELLOW [2nd]',
            group=self.color_group,
            command=lambda: self.select_color('Yellow'),
            width=215,
            border_color='#ffd814',
            fill_color='#ffffff',
            hover_color='#ffe24e'
        ).pack(side='left', padx=8)



        # ==================== ROW 2 - DIFFICULTY ==================== #
        self.create_header(self.root, 'DIFFICULTY')

        self.difficulty_frame = tk.Frame(self.root, bg=self.root['bg'])
        self.difficulty_frame.pack(pady=5)

        # button for easy difficulty
        Selectable_Round_Button(
            self.difficulty_frame,
            'EASY',
            group=self.difficulty_group,
            command=lambda: self.select_difficulty('Easy'),
            default_selected=True,
            border_color='#00c452',
            fill_color='#ffffff',
            hover_color='#00eb62'
        ).pack(side='left', padx=6)

        # button for medium difficulty
        Selectable_Round_Button(
            self.difficulty_frame,
            'MEDIUM',
            group=self.difficulty_group,
            command=lambda: self.select_difficulty('Medium'),
            border_color='#ffd814',
            fill_color='#ffffff',
            hover_color='#ffe24e'
        ).pack(side='left', padx=6)

        # button for hard difficulty
        Selectable_Round_Button(
            self.difficulty_frame,
            'HARD',
            group=self.difficulty_group,
            command=lambda: self.select_difficulty('Hard'),
            border_color='#e60000',
            fill_color='#ffffff',
            hover_color='#ff8181'
        ).pack(side='left', padx=6)



        # ==================== ROW 3 - FINALIZE CONFIG ==================== #
        self.create_header(self.root, 'READY? CLICK START')

        self.start_frame = tk.Frame(self.root, bg=self.root['bg'])
        self.start_frame.pack(pady=12)

        Selectable_Round_Button(
            self.start_frame,
            'START GAME',
            width=200,
            height=50,
            group=None,
            command=lambda: self.start_game(self.root)
        ).pack()

    def select_color(self, color):
        self.player_color = 'R' if color == 'Red' else 'Y'

    def select_difficulty(self, level):
        if level == 'Easy':
            self.ai_settings = {
                'mcts_simulations': 100,
                'mcts_max_depth': 6,
                'mcts_exploration_constant': 5
            }
        elif level == 'Medium':
            self.ai_settings = {
                'mcts_simulations': 300,
                'mcts_max_depth': 6,
                'mcts_exploration_constant': 3
            }
        elif level == 'Hard':
            self.ai_settings = {
                'mcts_simulations': 500,
                'mcts_max_depth': 8,
                'mcts_exploration_constant': 3
            }

    def start_game(self, root):
        root.destroy() # closes the window

    def create_header(self, parent, text):
        label = tk.Label(
            parent,
            text=text,
            font=('Arial', 12, 'bold'),
            bg=parent['bg']
        )
        label.pack(pady=(8, 2))



if __name__ == '__main__':
    # testing init config window
    w1 = Init_Config_Window()
    w1.root.mainloop()

    # seeing if we're able to extract data correctly
    print('DATA EXTRACTION')
    print(f"Chosen Opponent: {'Player' if w1.opponent_is_player else 'AI'}\n\n")
    
    
    
    
    # testing AI config window
    w2 = AI_Config_Window()
    w2.root.mainloop()

    # seeing if we're able to extract data correctly
    print('\n\nDATA EXTRACTION')
    print(f'Player Color: {w2.player_color}')
    print(f'AI Settings:\n{w2.ai_settings}')
    
    print('All good!')