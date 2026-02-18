# AlphaHorizons - A Connect-4 AlphaZero Algorithm From Scratch
by: Gabe Aquino

## Recreated the AlphaZero algorithm without external deep learning libraries. The following project includes:
* **neural network** implementation from scratch (including backpropagation)
* recreated **Monte-Carlo Tree Search (MCTS)** algorithm for guiding AI move search
* **database** implementation to store neural networks & game data for training
* **Connect-4 GUI** for the player to play against the AI (also has a gamemode for 2-player games)

## Main Inspiration
This project is moslty inspired by the AlphaGo paper by Deep Mind. Here is the link to the research paper for reference: https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf

## Cloning and Setup
### Below are the instructions to clone this repo and install any dependent python libraries to run the project.
1. Clone the repo into a directory of your choice (just copy and paste the command into the terminal)

```git clone https://github.com/al-gabe31/Connect_4_AlphaZero_from_Scratch.git```

2. Open the git repo in a code editor like VS Code (this one's optional but it's still nice to have it open)
3. Installing python libraries (you'll need these before actually running anything)

```pip install -r requirements.txt```

### If for some reason the ```pip install -r requirements.txt``` command doesn't work, you can just pip install the following python libraries one-by-one.
```
pip install matplotlib
pip install numpy
pip install pillow
```

## Running the Project
### All you need to do is execute the **run_application.py** file.
<img width="1620" height="497" alt="image" src="https://github.com/user-attachments/assets/7929b967-ac27-482e-9375-c6561489e901" />

### If everything works properly, a GUI window should pop up. Here, you should be able to choose whether you want to play against a player (or yourself), or against an AI that was trained.
<img width="743" height="281" alt="image" src="https://github.com/user-attachments/assets/2418e8c8-e014-4e68-bbab-7f618dcb9a4e" />

### If you select to play against an AI, another window should popup that allows you to customize the settings of the game. Here, you're able to select whether to go first or second, as well as the difficulty of the AI.
### One thing to note is that for higher difficulties, the AI will take more time to generate a move as a result of a more thorough search by the Monte-Carlo Tree Search.
<img width="709" height="477" alt="image" src="https://github.com/user-attachments/assets/436271a2-8a9c-45f5-a8c2-77ef27c21fea" />

### Once you have selected your settings, feel free to click "START GAME". For this example, I'll be playing yellow (going 2nd) with the easiest difficulty.

### If all goes well, you should now be able to play against the AI in a new window. Here is what the game would look like...
<img width="1570" height="927" alt="image" src="https://github.com/user-attachments/assets/9b79ac7e-36a4-4cd0-b7f5-06fa86f1f000" />

### Key
1. Visual representation of the game. To drop a piece, all you have to do is click on a column to drop a piece in that column.
  * Important Note: You can only place a piece once it's your turn.
2. Current turn description: let's you know who's turn it is (player or AI) and which piece that player controls (red or yellow)
3. History navigation settings: Allows you to review the game up to the current state. Feel free to replay the game move-by-move.
  * Important Note: You can only access the history navigation settings during your turn.
  * If you try to click on the navigation buttons while the AI is thinking, it'll tell you that the AI is currently thinking and to be patient.
4. Game logs: Helpful information on how the game is currently going. Here, you can see what move the AI chose and how long it took to come up with that move.
  * Important Note: Columns use 0 indexing
  * What that means is that "Move 0" just means the left-most column. "Move 1" is just the column to the right of it and "Move 6" is the right-most column.

### Trying to drop a piece or using the history navigation buttons during the AI's turn
As mentioned earlier, you're now allowed to drop a piece or use history navigation buttons during the AI's turn. You can easily tell if it's currently the AI's turn if you see "[AI]" at the bottom of the screen.
<img width="1562" height="910" alt="image" src="https://github.com/user-attachments/assets/d56e63f9-c30b-42f0-a8db-84a9f068d74e" />

* Doing so, you'll see the messages bracketed on the right. Notice how one of them has "29 / 100 simulations done" and the other "56 / 100 simulations done". This is more of an indicator as how close the AI is to finalizing their move.
* The more technical explanation to this is how far the AI is in their tree search to coming up with their next move.
* Once the AI has done 100 Mone-Carlo simulations, it'll then come up with its next move
* Here is how many simulations will run for each difficulty:
  * Easy: 100 simulations (pretty quick)
  * Medium: 300 Simulations (AI takes a little more time)
  * Hard: 500 simulations (AI will take its time to generate a move, so please be patient for this difficulty)
 
### End of the Game
Once the game is over, no player is able to place any more pieces (understandably). However, you can still use the history navigation buttons to review the game!
<img width="1550" height="908" alt="image" src="https://github.com/user-attachments/assets/f3ae1723-f2b6-4433-a8c7-cce8c07c528b" />

### Using the History Navigation Buttons
<img width="1561" height="912" alt="image" src="https://github.com/user-attachments/assets/6b8e36ac-096c-4f43-aa5e-b3fb39854239" />

### Key
1. Jump all the way to the beginning of the game
2. Move back 1 turn
3. Move forward 1 turn
4. Jump all the way to the current (or end) game state

## File Rundown
* config_window.py: code for the windows that pop up where you select your opponent and the AI settings
* create_db.ipynb: jupyter notebook file that was used to create the Connect_4.db and tables
* db_management.py: contains every function needed to interact with the Connect_4.db such as:
  * retrieving and storing neural networks
  * retrieving and storing self-play games for future training
  * running the training loop to train a neural network
  * viewing a past game
* evaluating_neural_network.ipynb: just a file to prove that the neural network works and can be used to learn a dataset
<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/a4cda5af-c550-4144-bb14-821535e89b26" />

* game_repr.py: contains the Game_State class to represent a particular game state (important for state-space search algorithms)
* game_window.py: code for the actual game window (the one where you drop pieces on the board and use the history navigation buttons)
* gui.py: old and outdated version of game_window.py (feel free to ignore)
* mcts.py: implementation for the Monte-Carlo Tree Search
* neural_network.py: my own version of a neural network from scratch without using exteral deep learning libraries
* notes.ipynb: just personal notes about Monte-Carlo Tree Search (feel free to ignore)
* requirements.txt: list of all dependencies you'll need to run the application (refer to Cloning and Setup section on what to do with this)
* run_application.py: the only file you really care about to run the game (just run this file and hopefully it starts the application without any issues)
* simulating_game.ipynb: just some old notes on how a game used to be simulated (the program uses functions to streamline the games instead)
* training_loop.ipynb: used to actually train and store neural networks (very important for the training aspect of the project!)
* src (folder): contains all visual assets of the game
* Connect_4.db (database): sqlite database containing information for the neural networks & game data

## Find an Issue?
Please reach out if you find anything wrong or have any suggestions for improvements! Any feedback is welcomed. Please contact me at alfonsoaquino100@gmail.com .
