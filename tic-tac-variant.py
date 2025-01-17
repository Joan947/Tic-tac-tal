#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COSC 4550-COSC5550 - Introduction to AI - Lab
"""
#-------------------------------------------------------------------------
# Tac-Tac-Tical
# NETID = jowusu1

# This program is designed to play Tic-Tac-Tical, using lookahead and board heuristics.
# It will allow the user to play a game against the machine, or allow the machine
# to play against itself for purposes of learning to improve its play.  All 'learning'
# code has been removed from this program.
#
# Tic-Tac-Tical is a 2-player game played on a grid. Each player has the same number
# of tokens distributed on the grid in an initial configuration.  On each turn, a player
# may move one of his/her tokens one unit either horizontally or vertically (not
# diagonally) into an unoccupied square.  The objective is to be the first player to get
# three tokens in a row, either horizontally, vertically, or diagonally.
#
# The board is represented by a matrix with extra rows and columns forming a
# boundary to the playing grid. Squares in the playing grid can be occupied by
# either 'X', 'O', or 'Empty' spaces.  The extra elements are filled with 'Out of Bounds'
# squares, which makes some of the computations simpler.
#-------------------------------------------------------------------------

from __future__ import print_function
import math
import random
from random import randrange
import copy
import time


def GetMoves (Player, Board):
#-------------------------------------------------------------------------
# Determines all legal moves for Player with current Board,
# and returns them in MoveList.
#-------------------------------------------------------------------------

	MoveList = []
	for i in range(1,NumRows+1):
		for j in range(1,NumCols+1):
			if Board[i][j] == Player:
			#-------------------------------------------------------------
			#  Check move directions (m,n) = (-1,0), (0,-1), (0,1), (1,0)
			#-------------------------------------------------------------
				for m in range(-1,2):
					for n in range(-1,2):
						if abs(m) != abs(n):
							if Board[i + m][j + n] == Empty:
								MoveList.append([i, j, i+m, j+n])

	return MoveList


def GetHumanMove (Player, Board):
#-------------------------------------------------------------------------
# If the opponent is a human, the user is prompted to input a legal move.
# Determine the set of all legal moves, then check input move against it.
#-------------------------------------------------------------------------
	MoveList = GetMoves(Player, Board)
	Move = None

	while(True):
		FromRow, FromCol, ToRow, ToCol = map(int, \
			input('Input your move (FromRow, FromCol, ToRow, ToCol): ').split(' '))

		ValidMove = False
		if not ValidMove:
			for move in MoveList:
				if move == [FromRow, FromCol, ToRow, ToCol]:
					ValidMove = True
					Move = move

		if ValidMove:
			break

		print('Invalid move.  ')

	return Move


def ApplyMove (Board, Move):
#-------------------------------------------------------------------------
# Perform the given move, and update Board.
#-------------------------------------------------------------------------

	FromRow, FromCol, ToRow, ToCol = Move
	newBoard = copy.deepcopy(Board)
	Board[ToRow][ToCol] = Board[FromRow][FromCol]
	Board[FromRow][FromCol] = Empty
	return Board


def InitBoard (Board):
#-------------------------------------------------------------------------
# Initialize the game board.
#-------------------------------------------------------------------------

	for i in range(0,BoardRows+1):
		for j in range(0,BoardCols+1):
			Board[i][j] = OutOfBounds

	for i in range(1,NumRows+1):
		for j in range(1,NumCols+1):
			Board[i][j] = Empty

	for j in range(1,NumCols+1):
		if odd(j):
			Board[1][j] = x
			Board[NumRows][j] = o
		else:
			Board[1][j] = o
			Board[NumRows][j] = x


def odd(n):
	return n%2==1

def ShowBoard (Board):
	print("")
	row_divider = "+" + "-"*(NumCols*4-1) + "+"
	print(row_divider)

	for i in range(1,NumRows+1):
		for j in range(1,NumCols+1):
			if Board[i][j] == x:
				print('| X ',end="")
			elif Board[i][j] == o:
				print('| O ',end="")
			elif Board[i][j] == Empty:
				print('|   ',end="")
		print('|')
		print(row_divider)

	print("")


def Win (Player, Board):
#-------------------------------------------------------------------------
# Determines if Player has won, by finding '3 in a row, col or diagonal'.
#-------------------------------------------------------------------------

	# Check rows for a win
	for i in range(1, NumRows + 1):
		for j in range(1, NumCols - 1):
			if Board[i][j] == Player and Board[i][j + 1] == Player and Board[i][j + 2] == Player:
				return True

	# Check columns for a win
	for i in range(1, NumRows - 1):
		for j in range(1, NumCols + 1	):
			if Board[i][j] == Player and Board[i + 1][j] == Player and Board[i + 2][j] == Player:
				return True

	# Check diagonals for a win (both directions)
	for i in range(1, NumRows - 1):
		for j in range(1, NumCols - 1):
			if Board[i][j] == Player and Board[i + 1][j + 1] == Player and Board[i + 2][j + 2] == Player:
				return True
			if Board[i][j + 2] == Player and Board[i + 1][j + 1] == Player and Board[i + 2][j] == Player:
				return True
	return False


# Heuristic (evaluation )function return a high value for board
# states that are favorable for Player, and a low value for board states that are unfavorable
def jowusu1_h(Player, Board):
    score = 0

    # Define the opponent
    opponent = o if Player == x else x

    # Reward for having two in a row with one empty space
    for i in range(1, NumRows + 1):
        for j in range(1, NumCols - 1):
            # Horizontal
            row = [Board[i][j], Board[i][j + 1], Board[i][j + 2]]
            if row.count(Player) == 2 and row.count(Empty) == 1:
                score += 10
            if row.count(opponent) == 2 and row.count(Empty) == 1:
                score -= 10  # Penalize for opponent's advantage

    # Reward for having two in a column with one empty space
    for i in range(1, NumRows - 1):
        for j in range(1, NumCols + 1):
            # Vertical
            col = [Board[i][j], Board[i + 1][j], Board[i + 2][j]]
            if col.count(Player) == 2 and col.count(Empty) == 1:
                score += 10
            if col.count(opponent) == 2 and col.count(Empty) == 1:
                score -= 10  # Penalize for opponent's advantage

    # Reward for having two in a diagonal with one empty space
    for i in range(1, NumRows - 1):
        for j in range(1, NumCols - 1):
            # Diagonal \
            if (Board[i][j] == Player and Board[i + 1][j + 1] == Player and Board[i + 2][j + 2] == Empty) or \
               (Board[i][j] == Player and Board[i + 1][j + 1] == Empty and Board[i + 2][j + 2] == Player) or \
               (Board[i][j] == Empty and Board[i + 1][j + 1] == Player and Board[i + 2][j + 2] == Player):
                score += 10
            if (Board[i][j] == opponent and Board[i + 1][j + 1] == opponent and Board[i + 2][j + 2] == Empty) or \
               (Board[i][j] == opponent and Board[i + 1][j + 1] == Empty and Board[i + 2][j + 2] == opponent) or \
               (Board[i][j] == Empty and Board[i + 1][j + 1] == opponent and Board[i + 2][j + 2] == opponent):
                score -= 10

            # Diagonal /
            if (Board[i][j + 2] == Player and Board[i + 1][j + 1] == Player and Board[i + 2][j] == Empty) or \
               (Board[i][j + 2] == Player and Board[i + 1][j + 1] == Empty and Board[i + 2][j] == Player) or \
               (Board[i][j + 2] == Empty and Board[i + 1][j + 1] == Player and Board[i + 2][j] == Player):
                score += 10
            if (Board[i][j + 2] == opponent and Board[i + 1][j + 1] == opponent and Board[i + 2][j] == Empty) or \
               (Board[i][j + 2] == opponent and Board[i + 1][j + 1] == Empty and Board[i + 2][j] == opponent) or \
               (Board[i][j + 2] == Empty and Board[i + 1][j + 1] == opponent and Board[i + 2][j] == opponent):
                score -= 10

    return score


 #Using minimax without alpha beta pruning

def minimax(Board, depth, maximizingPlayer, Player):
    global  states_eval_minmax,  max_depth_achieved_without_minmax

    # Track maximum depth achieved
    max_depth_achieved_without_minmax = max( max_depth_achieved_without_minmax, MaxDepth - depth)

    # Base case: terminal conditions
    if depth == 0 or Win(Player, Board) or Win(o if Player == x else x, Board):
        states_eval_minmax += 1  # Count this state
        return jowusu1_h(Player, Board)

    if maximizingPlayer:
        maxEval = -infinity
        for move in GetMoves(Player, Board):
            newBoard = ApplyMove(copy.deepcopy(Board), move)
            eval = minimax(newBoard, depth - 1, False, Player)
            maxEval = max(maxEval, eval)
        return maxEval
    else:
        minEval = infinity
        opponent = o if Player == x else x
        for move in GetMoves(opponent, Board):
            newBoard = ApplyMove(copy.deepcopy(Board), move)
            eval = minimax(newBoard, depth - 1, True, Player)
            minEval = min(minEval, eval)
        return minEval

# Using minimax with alpha beta pruning and heuristic evaluation function
def minimax_with_pruning(Board, depth, alpha, beta, maximizingPlayer, Player):
    global states_eval, max_depth_achieved

    # Track maximum depth achieved
    max_depth_achieved = max(max_depth_achieved, MaxDepth - depth)

    # Base case: terminal conditions
    if depth == 0 or Win(Player, Board) or Win(o if Player == x else x, Board):
        states_eval += 1  # Count this state
        return jowusu1_h(Player, Board)

    if maximizingPlayer:
        maxEval = -infinity
        for move in GetMoves(Player, Board):
            newBoard = ApplyMove(copy.deepcopy(Board), move)
            eval = minimax_with_pruning(newBoard, depth - 1, alpha, beta, False, Player)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Alpha-beta pruning
        return maxEval
    else:
        minEval = infinity
        opponent = o if Player == x else x
        for move in GetMoves(opponent, Board):
            newBoard = ApplyMove(copy.deepcopy(Board), move)
            eval = minimax_with_pruning(newBoard, depth - 1, alpha, beta, True, Player)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha-beta pruning
        return minEval


# Using Monte Carlo simulation
# using 100 simulations
def MonteCarloSimulation(Board, Player, simulations=100, max_depth=10):
    explored_states = 0
    max_depth_reached = 0

    def rollout(state, player, current_depth):
        """Perform a simulation from the given state."""
        nonlocal explored_states, max_depth_reached
        while current_depth < max_depth and not Win(player, state) and not Win(-player, state):
            moves = GetMoves(player, state)
            if not moves:
                break
            explored_states += 1
            max_depth_reached = max(max_depth_reached, current_depth)
            state = ApplyMove(state, random.choice(moves))
            player = -player
            current_depth += 1
        if Win(player, state):
            return 1
        elif Win(-player, state):
            return -1
        return 0

    best_move = None
    best_score = -infinity
    for move in GetMoves(Player, Board):
        move_score = 0
        for _ in range(simulations):
            move_score += rollout(ApplyMove(copy.deepcopy(Board), move), -Player, 1)
        if move_score > best_score:
            best_score = move_score
            best_move = move

    return best_move, explored_states, max_depth_reached


# Using expectiminimax
def expectiminimax(Board, depth, alpha, beta, maximizingPlayer, Player):

    global states_eval_expected_minimax, max_depth_achieved_expected_minimax

    # Track maximum depth achieved
    max_depth_achieved_expected_minimax = max(max_depth_achieved_expected_minimax, MaxDepth - depth)

    # Base case: terminal condition or maximum depth
    if depth == 0 or Win(x, Board) or Win(o, Board) or not (GetMoves(x, Board) or GetMoves(o, Board)):
        states_eval_expected_minimax += 1
        return jowusu1_h(Player, Board), None

    if maximizingPlayer:
        # Maximizing player's turn
        max_value = -float('inf')
        best_move = None
        for move in GetMoves(Player, Board):
            simulated_board = ApplyMove(copy.deepcopy(Board), move)
            value, _ = expectiminimax(simulated_board, depth - 1, alpha, beta, False, o if Player == x else x)
            if value > max_value:
                max_value = value
                best_move = move
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # Alpha-beta pruning
        return max_value, best_move

    else:
        # Chance node: Expectation over all opponent's moves
        total_value = 0
        possible_moves = GetMoves(Player, Board)
        if not possible_moves:  # No moves available
            return jowusu1_h(Player, Board), None

        for move in possible_moves:
            simulated_board = ApplyMove(copy.deepcopy(Board), move)
            value, _ = expectiminimax(simulated_board, depth - 1, alpha, beta, True, o if Player == x else x)
            total_value += value / len(possible_moves)

        return total_value, None



# Q-learning AI setup


def GetComputerMove(Player, Board):
	#-------------------------------------------------------------------------
	# If the opponent is a computer, use artificial intelligence to select
	# the best move.
	# For this demo, a move is chosen at random from the list of legal moves.
	# You need to write your own code to get the best computer move.
	#-------------------------------------------------------------------------
	#Computing for the best move using minmax, alpha beta pruning, expectiminimax and monte carlo simulation
	global states_eval, max_depth_achieved
	global  states_eval_minmax,  max_depth_achieved_without_minmax
	global states_eval_expected_minimax, max_depth_achieved_expected_minimax
	
	states_eval = 0
	max_depth_achieved = 0

	states_eval_minmax = 0
	max_depth_achieved_without_minmax = 0

	states_eval_expected_minimax = 0
	max_depth_achieved_expected_minimax = 0
	
	bestMove = None
	bestMoveNoPruning = None
	bestMoveExpectiMinimax = None
	bestValue = -infinity
	bestValue_no_pruning = -infinity
	bestValue_with_expected_minimax = -infinity

	m_c_best_move = None

	

	start_time_minmax= time.time()
	for move in GetMoves(Player, Board):
		newBoard = ApplyMove(copy.deepcopy(Board), move)
		moveValue = minimax(newBoard, MaxDepth, False, Player)
		if moveValue > bestValue_no_pruning:
			bestValue_no_pruning = moveValue
			bestMoveNoPruning = move
	elapsed_time_with_no_pruning = time.time() - start_time_minmax
      


	# Using Alpha beta pruning
	start_time = time.time()
	for move in GetMoves(Player, Board):
		newBoard = ApplyMove(copy.deepcopy(Board), move)
		moveValue = minimax_with_pruning(newBoard, MaxDepth, -infinity, infinity, False, Player)
		if moveValue > bestValue:
			bestValue = moveValue
			bestMove = move
	
	elapsed_time = time.time() - start_time

	# Using Monte Carlo Tree Search (MCTS)
	m_c_start_time = time.time()
	m_c_best_move, m_c_explored_states, m_c_max_depth = MonteCarloSimulation(Board, Player, max_depth=MaxDepth)
	m_c_elapsed_time = time.time() - m_c_start_time

	start_time_expected_minimax = time.time()
	for move in GetMoves(Player, Board):
		newBoard = ApplyMove(copy.deepcopy(Board), move)
		moveValue,_ = expectiminimax(newBoard, MaxDepth, -infinity, infinity, False, Player)
		if moveValue > bestValue_with_expected_minimax:
			bestValue_with_expected_minimax = moveValue
			bestMoveExpectiMinimax = move
	
	elapsed_time_expected_minimax = time.time() - start_time_expected_minimax

	
	print(f"States Evaluated with pruning: {states_eval}")
	print(f"Max Depth Achieved with pruning: {max_depth_achieved}")
	print(f"Time Taken with pruning: {elapsed_time:.4f} seconds\n\n")

	print(f"States Evaluated without pruning: { states_eval_minmax}")
	print(f"Max Depth Achieved without pruning: { max_depth_achieved_without_minmax}")
	print(f"Time Taken without pruning: {elapsed_time_with_no_pruning:.4f} seconds\n\n")

	print(f"States Evaluated with expected minimax: {states_eval_expected_minimax}")
	print(f"Max Depth Achieved with expected minimax: {max_depth_achieved_expected_minimax}")
	print(f"Time Taken with expected minimax: {elapsed_time_expected_minimax:.4f} seconds\n\n")

	print(f"Monte Carlo Playouts: {m_c_explored_states}")
	print(f"Monte Carlo Max Depth Reached: {m_c_max_depth}")
	print(f"Monte Carlo Time Taken: {m_c_elapsed_time:.4f} seconds\n")





	# Find the minimum elapsed time among the three algorithms and select best move from minimum time
	min_time = min(elapsed_time, elapsed_time_expected_minimax, m_c_elapsed_time)

	# Return the best move based on the minimum elapsed time
	if min_time == elapsed_time:
		print("Selected Minimax with pruning")
		return bestMove
	elif min_time == elapsed_time_expected_minimax:
		print("Selected Expectiminimax")
		return bestMoveExpectiMinimax
	else:
		print("Selected Monte Carlo")
		return m_c_best_move
      

	# return bestMove












if __name__ == "__main__":
#-------------------------------------------------------------------------
# A move is represented by a list of 4 elements, representing 2 pairs of
# coordinates, (FromRow, FromCol) and (ToRow, ToCol), which represent the
# positions of the piece to be moved, before and after the move.
#-------------------------------------------------------------------------
	x = -1
	o = 1
	Empty = 0
	OutOfBounds = 2
	NumRows = 5
	BoardRows = NumRows + 1
	NumCols = 4
	BoardCols = NumCols + 1
	MaxMoves = 4*NumCols
	NumInPackedBoard = 4 * (BoardRows+1) *(BoardCols+1)
	infinity = 10000  # Value of a winning board
	MaxDepth = 4
	Board = [[0 for col in range(BoardCols+1)] for row in range(BoardRows+1)]
   

	print("\nThe squares of the board are numbered by row and column, with '1 1' ")
	print("in the upper left corner, '1 2' directly to the right of '1 1', etc.")
	print("")
	print("Moves are of the form 'i j m n', where (i,j) is a square occupied")
	print("by your piece, and (m,n) is the square to which you move it.")
	print("")


	

	InitBoard(Board)
	ShowBoard(Board)

	MoveList = GetMoves(x,Board)
	print(MoveList)
	MoveList = GetMoves(o,Board)
	print(MoveList)
	players = ["human","program"]
	first_player = random.choice(players)

	tokens = [x,o]
	# choosing at random the token for the players
	human_token = random.choice(tokens)

	if human_token == x:
		print("You move the 'X' pieces.\n")
		computer_token = o
	else: 
		computer_token = x
		print("You move the 'o' pieces.\n")


	## Allowing human or program to be either first or second player
	if first_player == "human":
		while True:
			Move = GetHumanMove(human_token,Board)
			Board = ApplyMove(Board,Move)
			ShowBoard(Board)
			if Win(human_token,Board):
				print("Hurray!!, You have won the victory !")
				break
			Move = GetComputerMove(computer_token,Board)
			Board = ApplyMove(Board,Move)
			ShowBoard(Board)
			if Win(computer_token,Board):
				print("Awwch!!, Computer won,You have lost!")
				break
	else:
		while True:
			Move = GetComputerMove(computer_token,Board)
			Board = ApplyMove(Board,Move)
			ShowBoard(Board)
			if Win(computer_token,Board):
				print("Awwch!!, Computer won,You have lost !")
				break
			Move = GetHumanMove(human_token,Board)
			Board = ApplyMove(Board,Move)
			ShowBoard(Board)
			if Win(human_token,Board):
				print("Hurray!!, You have won the victory !")
				break
