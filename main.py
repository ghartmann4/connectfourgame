import gameplay

# increase depth here to improve the computer player's game performance
# recommended range: 4 - 7 (higher depths take longer to run)
MINIMAX_DEPTH = 5
NUM_ROWS = 6
NUM_COLUMNS = 7

game = gameplay.Game(num_rows=NUM_ROWS,num_columns=NUM_COLUMNS, minimax_depth=MINIMAX_DEPTH)
game.run_game()