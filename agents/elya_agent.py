from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("elya_agent")
class ElyaAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(ElyaAgent, self).__init__()
        self.name = "ElyaAgent"

    def step(self, chess_board, player, opponent):
        """
        Implements the step function of your agent, determining the next move
        based on the current game phase.
        """
        start_time = time.time()
        valid_moves = get_valid_moves(chess_board, player)

        if not valid_moves:
            # Pass if no valid moves are available
            return None

        # Determine the current game phase
        num_empty_cells = np.sum(chess_board == 0)
        if num_empty_cells > 40:  # Opening phase
            move = self.opening_phase(chess_board, player, valid_moves)
        elif num_empty_cells > 10:  # Midgame phase
            move = self.midgame_phase(chess_board, player, valid_moves, opponent)
        else:  # Endgame phase
            move = self.endgame_phase(chess_board, player, valid_moves, opponent)

        time_taken = time.time() - start_time
        print(f"My AI's turn took {time_taken:.4f} seconds.")

        return move

    def opening_phase(self, board, player, valid_moves):
        """
        Select a move prioritizing corners and maximizing mobility in the opening phase.
        """
        # Define corners
        corners = [(0, 0), (0, board.shape[1] - 1),
                   (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]

        # Prioritize corners if available
        for move in valid_moves:
            if move in corners:
                return move

        # Otherwise, maximize mobility
        return max(valid_moves, key=lambda m: len(get_valid_moves(execute_move(board.copy(), m, player), player)))

    def midgame_phase(self, board, player, valid_moves, opponent):
        """
        Use heuristic evaluation to select a move during the midgame phase.
        """
        def evaluate_move(move):
            if move not in valid_moves:
                return float('-inf')  # Safeguard against invalid moves

            temp_board = execute_move(board.copy(), move, player)
            if temp_board is None:
                print(f"Invalid move detected: {move}")
                return float('-inf')  # Safeguard against execution errors

            opponent_moves = len(get_valid_moves(temp_board, opponent))
            stable_discs = self.count_stable_discs(temp_board, player)
            return -opponent_moves + stable_discs

        # Choose the move with the best heuristic score
        return max(valid_moves, key=evaluate_move)

    def endgame_phase(self, board, player, valid_moves, opponent):
        """
        Focus on maximizing captures and stability in the endgame phase.
        """
        # Evaluate moves based on the number of discs captured
        return max(valid_moves, key=lambda m: count_capture(board, m, player))

    def count_stable_discs(self, board, player):
        """
        Count the number of stable discs for the given player.
        """
        stable_count = 0
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                if board[r, c] == player and self.is_stable(board, (r, c), player):
                    stable_count += 1
        return stable_count

    def is_stable(self, board, position, player):
        """
        Determine if a disc at the given position is stable.
        """
        rows, cols = board.shape
        r, c = position

        # Simplified assumption: consider edge or corner discs stable
        if r in {0, rows - 1} or c in {0, cols - 1}:
            return True

        # Additional stability checks can be added for more precision
        return False
