from enum import Enum
import copy 
import pygame
import sys

class GameOver(Enum):
    NOT_OVER = 0
    DRAWN = 1
    WIN_FOR_1 = 2
    WIN_FOR_2 = 3

class Player_ID(Enum):
    PLAYER_1 = 1
    PLAYER_2 = 2

class Board():
    def __init__(self, num_rows, num_columns) -> None:
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.board = [ [0]*num_columns for _ in range(num_rows)]
        self.move_number = 1

    def __str__(self) -> str:
        lines = []
        for row in self.board:
            lines.append(str(row))
        return "\n".join(lines)
    
    def set_board_position(self, array, move_number):
    """For testing purposes: set board position to a passed-in array.
    Args:
        array: the board position to be set
        move_number: the move number to be set

    Returns:
        none, updates boards position
    """
        self.num_rows = len(array)
        self.num_columns = len(array[0])
        self.board = array
        self.move_number = move_number
    
    def is_valid_move(self, target_column):
        """For testing purposes: set board position to a passed-in array.
        Args:
            target_column: the column index to check

        Returns:
            True, if that column has an empty space for a piece to be placed
            False, otherwise
        """
        if target_column<0 or target_column >=self.num_columns:
            return False
        elif self.board[0][target_column] != 0:
            return False
        else:
            return True
        
    def next_player_to_move(self):
        if self.move_number % 2 == 1:
            return Player_ID.PLAYER_1
        else:
            return Player_ID.PLAYER_2
        
    def last_player_that_moved(self):
        if self.move_number % 2 == 1:
            return Player_ID.PLAYER_2
        else:
            return Player_ID.PLAYER_1
        
    def get_piece(self,row,column):
        return self.board[row][column]
    
    def place_piece(self, column):
        """ Places piece in column
        Args:
            column: the column index to place the piece

        Returns:
            Nothing, places the piece in target column if possible.
            If the move is invalid, it does not place a piece.
        """
        if self.next_player_to_move() == Player_ID.PLAYER_1:
            piece = 1
        else:
            piece = 2

        if not self.is_valid_move(column):
            return
        
        row = self.lowest_unoccupied_row(column)
        self.board[row][column] = piece
        self.move_number += 1
        return
        
    def lowest_unoccupied_row(self, column):
        """ Returns row index of lowest open slot in target column, or returns -1 if that column is full
        Args:
            column: the column index to place the piece
        """
        for row in range(self.num_rows-1,-1,-1):
                if self.get_piece(row, column) == 0:
                    return row
        return -1

    def game_over_eval(self):
        """ Checks if game is over
        Returns:
            Status of game- GameOver.WIN_FOR_1, GameOver.WIN_FOR_2, 
                            GameOver.DRAWN, or GameOver.NOT_OVER
        """
    
        pos_diagonals = self.get_positive_diagonals()
        neg_diagonals = self.get_negative_diagonals()
        columns = self.get_columns()
        rows = self.board

        directions = [pos_diagonals, neg_diagonals, rows, columns]

        for direction in directions:
            for line in direction:
                eval = self.contains_4_in_a_row(line)
                if eval == GameOver.WIN_FOR_1:
                    return GameOver.WIN_FOR_1
                elif eval == GameOver.WIN_FOR_2:
                    return GameOver.WIN_FOR_2
                else:
                    continue
        
        if self.move_number > self.num_rows * self.num_columns:
            return GameOver.DRAWN
        
        return GameOver.NOT_OVER
  
    def contains_4_in_a_row(self, line):
        """ Checks if given line contains 4 in a row
        Args:
            line - an 1-dimensional array representing one row, column, or diagonal of the board

        Returns:
            If connect4 exists: GameOver.WIN_FOR_1 or GameOver.WIN_FOR_2
            Else, returns GameOver.NOT_OVER
        """
        if len(line) <=3: 
            return GameOver.NOT_OVER
        last_spot = 0
        for idx, spot in enumerate(line):
            if idx == 0:
                last_spot = spot
                if spot == 0:
                    streak_length = 0
                else:
                    streak_length = 1
            else:
                if spot == 0:
                    last_spot = spot
                    streak_length = 0
                    continue

                elif spot == last_spot:
                    streak_length += 1
                    if streak_length >= 4:
                        if spot == 1:
                            return GameOver.WIN_FOR_1
                        else:
                            return GameOver.WIN_FOR_2
                else:
                    last_spot = spot
                    streak_length = 1
        return GameOver.NOT_OVER

    def get_positive_diagonals(self):

        diagonals = []

        for row_index in range(self.num_rows):
            row = row_index
            current_diagonal = []
            col = 0
            while 0 <= row < self.num_rows and 0 <= col < self.num_columns:
                current_diagonal.append(self.board[row][col])
                row -= 1
                col += 1
            diagonals.append(current_diagonal)

        for col_index in range(1, self.num_columns):
            col = col_index
            row = self.num_rows-1
            current_diagonal = []
            while 0 <= row < self.num_rows and 0 <= col < self.num_columns:
                current_diagonal.append(self.board[row][col])
                row -= 1
                col += 1
            diagonals.append(current_diagonal)

        return diagonals

    def get_negative_diagonals(self):

        diagonals = []

        for row_index in range(self.num_rows-1,-1,-1):
            row = row_index
            current_diagonal = []
            col = 0
            while 0 <= row < self.num_rows and 0 <= col < self.num_columns:
                current_diagonal.append(self.board[row][col])
                row += 1
                col += 1
            diagonals.append(current_diagonal)

        for col_index in range(1, self.num_columns):
            col = col_index
            row = 0
            current_diagonal = []
            while 0 <= row < self.num_rows and 0 <= col < self.num_columns:
                current_diagonal.append(self.board[row][col])
                row += 1
                col += 1
            diagonals.append(current_diagonal)

        return diagonals

    def get_columns(self):
        columns = []
        for col in range(self.num_columns):
            current_column = []
            for row in range(self.num_rows):
                current_column.append(self.board[row][col])
            columns.append(current_column)
        return columns
    
class Game():
    def __init__(self, num_rows, num_columns, minimax_depth=4) -> None:
        self.board = Board(num_rows, num_columns)
        self.move_number = 1
        self.winner = None
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.minimax_depth = minimax_depth

        self.evaluation_table = self.make_evaluation_table()
        
        self.SQUARESIZE = 700//num_columns
        self.RADIUS = int(self.SQUARESIZE/2 - 5)
        self.BLUE = (0,0,255)
        self.LIGHT_BLUE = (0,100,255)
        self.BLACK = (0,0,0)
        self.RED = (255,0,0)
        self.DARK_RED = (200, 0, 0)
        self.YELLOW = (255,255,0)
        self.DARK_YELLOW = (230, 230, 0)
        
        pygame.init()
        self.width = self.num_columns * self.SQUARESIZE
        self.height = (self.num_rows+1) * self.SQUARESIZE
        size = (self.width, self.height)
        self.screen = pygame.display.set_mode(size)
        self.myFont = pygame.font.SysFont("couriernew", 65)

    def update_winner(self, winner):
        self.winner = winner
        return
    
    def make_evaluation_table(self):
        """ Creates a new row x column array used to evaluate a position
        Each spot is given a value based on the number of connect 4 lines it can participate in.
        """
        table = [[0]*self.num_columns for _ in range(self.num_rows)]
        for i in range(self.num_rows):
            for j in range(self.num_columns):
                num_connect_4 = 0
                # count possible vertical connect-4s for this index
                upper_bound = max(0, i - 3)
                lower_bound = min(self.num_rows-1, i + 3)
                verticals = max(0, (lower_bound-upper_bound)-2)
                num_connect_4 += verticals

                # count possible horizontal connect-4s for this index
                left_bound = max(0, j - 3)
                right_bound = min(self.num_columns-1, j + 3)
                horizontals = max(0, (right_bound-left_bound)-2)
                num_connect_4 += horizontals

                # count possible SW - NE connect-4s for this index
                distance_to_left_bound = j - left_bound
                distance_to_lower_bound = lower_bound - i
                southwest_bound = min(distance_to_left_bound, distance_to_lower_bound)
                distance_to_right_bound = right_bound - j
                distance_to_upper_bound = i - upper_bound
                northeast_bound = min(distance_to_right_bound, distance_to_upper_bound)
                lower_L_to_upper_R_connect4 = max(0, (southwest_bound+northeast_bound)-2)
                num_connect_4 += lower_L_to_upper_R_connect4

                # count possible NW - SE connect-4s for this index
                northwest_bound = min(distance_to_left_bound, distance_to_upper_bound)
                southeast_bound = min(distance_to_right_bound, distance_to_lower_bound)
                upper_R_to_lower_L_connect4 = max(0, (northwest_bound+southeast_bound)-2)
                num_connect_4 += upper_R_to_lower_L_connect4

                table[i][j] = num_connect_4
    
        return table
    
    def make_move(self, target_column):
        """ Makes move on board
        Args:
            target_column: column move is to be played in

        Returns:
            None; plays move on board, checks if game is over.
        """
        if self.board.is_valid_move(target_column):
            # play move in board
            self.board.place_piece(target_column)

            eval = self.board.game_over_eval()
            if eval == GameOver.NOT_OVER:
                return
            elif eval == GameOver.WIN_FOR_1:
                self.update_winner(winner=GameOver.WIN_FOR_1)
                return
            elif eval == GameOver.WIN_FOR_2:
                self.update_winner(winner=GameOver.WIN_FOR_2)
                return
            elif eval == GameOver.DRAWN:
                self.update_winner(winner=GameOver.DRAWN)
                return
            else:
                return
        else:
            return
    
    def get_computer_move(self):
        eval, move = self.minimax(self.board, alpha=-float("inf"), beta=float("inf"), first_call=True)
        return move
    
    def estimate_position_eval(self, board):
        """ Estimates position using evaluation table
        Args:
            board: a hypothetical board position

        Returns:
            score: an integer that estimates the whether the position is better for player 1 (positive) or 
            player 2 (negative), and an estimate of how "winning" it is (magnitude)

            score is the sum of the evaluation_table values of player 1's pieces minus
            the sum of the evaluation_table values of player 2's pieces
        """
        score = 0
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                if board.board[row][col] == 1:
                    score += self.evaluation_table[row][col]
                elif board.board[row][col] == 2:
                    score -= self.evaluation_table[row][col]
        return score

        
    def set_game_position(self, board=Board):
        # for testing purposes
        self.board = board
        self.winner = None
        self.num_rows = board.num_rows
        self.num_columns = board.num_columns
        self.minimax_memo = {}
        return
    
    def draw_black_bar_at_top(self):
        # cover up text or circle that appeared in top bar
        pygame.draw.rect(self.screen, self.BLACK, ((0,0, self.width, self.SQUARESIZE))) 
        pygame.display.update()
        return 
    
    def run_game(self):
        
        self.draw_board(self.screen)

        while self.winner is None and self.board.move_number <= self.board.num_rows * self.board.num_columns:
            if self.board.next_player_to_move() == Player_ID.PLAYER_1:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    elif event.type == pygame.MOUSEMOTION:
                        self.draw_black_bar_at_top()
                        posx = event.pos[0]
                        pygame.draw.circle(self.screen, self.RED, (posx, self.SQUARESIZE//2), self.RADIUS)
                        pygame.draw.circle(self.screen, self.DARK_RED, (posx, self.SQUARESIZE//2), int(.8*self.RADIUS))
                        pygame.display.update()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        self.draw_black_bar_at_top()
                        posx = event.pos[0]
                        column = (posx//self.SQUARESIZE)
                        self.make_move(column)


            else:
                self.draw_black_bar_at_top()
                computer_turn_message = self.myFont.render("Computer thinking", 1, self.LIGHT_BLUE)
                self.screen.blit(computer_turn_message, (20,10))
                pygame.display.update()
                column = self.get_computer_move()
                self.make_move(column)
                self.draw_black_bar_at_top()
                pygame.event.clear()

            self.draw_board(self.screen)

        if self.winner == GameOver.WIN_FOR_1:
            game_over_message = self.myFont.render("Player 1 Wins!", 1, self.RED)
        elif self.winner == GameOver.WIN_FOR_2:
            game_over_message = self.myFont.render("Player 2 Wins!", 1, self.YELLOW)
        else:
            game_over_message = self.myFont.render("Game drawn!", 1, self.BLUE)

        self.draw_black_bar_at_top()
        self.screen.blit(game_over_message, (40,10))
        pygame.display.update()
        pygame.time.wait(10000)

        return
    
    def draw_board(self, screen):
        for col in range(self.num_columns):
            for row in range(self.num_rows):
                pygame.draw.rect(screen, self.BLUE, (col*self.SQUARESIZE, row*self.SQUARESIZE+self.SQUARESIZE, self.SQUARESIZE, self.SQUARESIZE))
                if self.board.board[row][col] == 1:
                    self.draw_circle(row, col, color=self.RED, inner_color=self.DARK_RED)
                elif self.board.board[row][col]==2:
                    self.draw_circle(row, col, color=self.YELLOW, inner_color=self.DARK_YELLOW)
                else:
                    self.draw_circle(row, col, color=self.BLACK)

        pygame.display.update()
        return

    def draw_circle(self, row, col, color, inner_color=None):

        pygame.draw.circle(self.screen, color, (col*self.SQUARESIZE+self.SQUARESIZE//2, row*self.SQUARESIZE+self.SQUARESIZE+self.SQUARESIZE//2), self.RADIUS) 
        if inner_color:
            pygame.draw.circle(self.screen, inner_color, (col*self.SQUARESIZE+self.SQUARESIZE//2, row*self.SQUARESIZE+self.SQUARESIZE+self.SQUARESIZE//2), int(.8*self.RADIUS)) 
        return

    def minimax(self, board: Board, alpha, beta, depth = None, first_call=False):
        """ Minimax algorithm to choose a move
        Args:
            board: a given board position
            alpha: evaluation to be used in alpha-beta pruning
            beta: evaluation to be used in alpha-beta pruning
            depth: the remaining number of recursive levels to check
            first_call: whether this is the first call of the algorithm

        Returns:
            evaluation, move: the best move according to the algorithm and its corresponding evaluation
        """

        if depth is None:
            depth = self.minimax_depth

        candidate_moves = []

        # first, check to see if there is a winning move, if there is, then return that without considering other moves
        for move in range(board.num_columns):

            if board.is_valid_move(move):
                new_board = copy.deepcopy(board)
                new_board.place_piece(move)

                end_condition = new_board.game_over_eval()
                if end_condition == GameOver.WIN_FOR_1 or end_condition == GameOver.WIN_FOR_2:
                    if end_condition == GameOver.WIN_FOR_1:
                        evaluation = 9999
                    else:
                        evaluation = -9999

                    return [evaluation, move]
                
                elif end_condition == GameOver.DRAWN:
                    candidate_moves.append((move, GameOver.DRAWN))
                else:
                    candidate_moves.append((move, GameOver.NOT_OVER))

        if depth <= 0:
            return [self.estimate_position_eval(board),-1]

        # otherwise, go through each candidate move one by one, and call the minimax algorithm
        # return the candidate move with the highest evaluation (if player 1) or the lowest evaluation (if player 2)
        
        if board.next_player_to_move() == Player_ID.PLAYER_1:
            # play for MAX eval
            maxEval = [-float("inf"), -1]
            for move, draw_condition in candidate_moves:

                new_board = copy.deepcopy(board)
                new_board.place_piece(move)

                if draw_condition == GameOver.DRAWN:
                    evaluation = (0, move)
                else:
                    evaluation = self.minimax(new_board, alpha, beta, depth-1)
                
                if evaluation[0] > maxEval[0]:
                    maxEval = [evaluation[0], move]
                alpha = max(alpha, evaluation[0])
                if beta <= alpha:
                    break

            return maxEval

        else:
            # play for MIN eval
            minEval = [float("inf"), -1]
            for move, draw_condition in candidate_moves:

                if first_call:
                    self.draw_black_bar_at_top()
                    self.draw_circle(-1, move, color=self.YELLOW, inner_color=self.DARK_YELLOW)
                    computer_turn_message = self.myFont.render("Computer thinking", 1, self.LIGHT_BLUE)
                    self.screen.blit(computer_turn_message, (20,10))
                    pygame.display.update()

                new_board = copy.deepcopy(board)
                new_board.place_piece(move)

                if draw_condition == GameOver.DRAWN:
                    evaluation = (0, move)
                else:
                    evaluation = self.minimax(new_board, alpha, beta, depth-1)
                    
                if evaluation[0] < minEval[0]:
                    minEval = [evaluation[0], move]
                beta = min(beta, evaluation[0])
                if beta <= alpha:
                    break

            return minEval


game = Game(6,7,4)
