from gosgf import Sgf_game
from go_core.goboard import GoBoard

import numpy as np
import mxnet as mx

class DataGenerator(object):
    def __init__(self, sgf_file, batch_size=30, history_length=8):
        self.sgf_file = sgf_file
        self.board_col = 19
        self.board_row = 19
        self.history_length = history_length
        self.board_length = self.board_row * self.board_col

        self.go_board = GoBoard(self.board_col)
        self.data_b = np.zeros((self.history_length, self.board_row, self.board_col))
        self.data_w = np.zeros((self.history_length, self.board_row, self.board_col))
        self.label = 0 # np.zeros((self.board_row * self.board_col))

        sgf_content = open(sgf_file,'r').read()
      
        sgf = Sgf_game.from_string(sgf_content)

        self.main_sequence_iter = sgf.main_sequence_iter()
        
        if sgf.get_handicap() != None and sgf.get_handicap() != 0:
          for setup in sgf.get_root().get_setup_stones():
            for move in setup:
              self.go_board.apply_move('b', move)

        # get current state as the t-1 state, will be all zero if there is no handicap
        
        

    def next(self):

      item = self.main_sequence_iter.next()
      color, move = item.get_move()
      
      while color is None or move is None:
        item = self.main_sequence_iter.next()
        color, move = item.get_move()

      row, col = move
      self.label = np.zeros((self.board_row * self.board_col))
      self.label = self.board_row*row + col
      for i in range(self.history_length-1):
        self.data_b[i] = self.data_b[i+1]
        self.data_w[i] = self.data_w[i+1]

      self.data_b[-1] = self.go_board.get_state('b')
      self.data_w[-1] = self.go_board.get_state('w')

      self.go_board.apply_move(color, move)

      if color == 'b':
        return self.data_b, self.label
      else:
        return self.data_w, self.label
      # print(str(color)+':'+str(row) + ',' + str(col))

        

       
        

