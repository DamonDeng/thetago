from gosgf import Sgf_game
from go_core.goboard import GoBoard

import numpy as np
import mxnet as mx

class OriginalProcessor(object):

    def __init__(self, sgf_file, board_size=19, level_limit='0d', player='all'):
        self.sgf_file = sgf_file
        self.board_col = board_size
        self.board_row = board_size
        self.level_limit = level_limit # the value should be nk, nd, np, n is the level number
        self.player = player # the value should be 'all', 'winner' or 'loser'
        self.board_length = self.board_row * self.board_col

        self.go_board = GoBoard(self.board_col)
        
        self.label = 0 # np.zeros((self.board_row * self.board_col))

        sgf_content = open(sgf_file,'r').read()
      
        self.sgf = Sgf_game.from_string(sgf_content)

        self.main_sequence_iter = self.sgf.main_sequence_iter()
        
        if self.sgf.get_handicap() != None and self.sgf.get_handicap() != 0:
          # print('handling handicap')
          for setup in self.sgf.get_root().get_setup_stones():
            for move in setup:
              self.go_board.apply_move('b', move)


    @classmethod
    def get_processor(cls, sgf_file, board_size=19, level_limit='0d', player='all'):
      return cls(sgf_file, board_size, level_limit, player)

    @classmethod
    def get_data_shape_only(cls, batch_size):
      board_col = 19
      board_row = 19
      return (batch_size, 7, board_row, board_col)

    @classmethod
    def get_label_shape_only(cls, batch_size):
      return (batch_size, )

    @classmethod
    def get_data_shape(cls, batch_size):
      board_col = 19
      board_row = 19
      return [('data',(batch_size, 7, board_row, board_col))]

    @classmethod
    def get_label_shape(cls, batch_size):
      return [('softmax_label', (batch_size, ))]


    def get_generator(self):

      # kind of urgly way to select game played by player with d or p level
      if self.sgf.is_level_higher_than(self.level_limit):

        winner = self.sgf.get_winner()


        for item in self.main_sequence_iter:

          color, move = item.get_move()

          is_target = False
          if self.player == 'all':
            is_target = True
          elif self.player == 'winner':
            if color == winner:
              is_target = True
            else:
              is_target = False
          elif self.player == 'loser':
            if color == winner:
              is_target = False
            else:
              is_target = True
          else:
            # if the player is other than winner and loser, 
            # maybe the developer call this function didn't make it right, just get all the moves.
            is_target = True
            
          
          if not color is None and not move is None:

            # print('-------------------------------')
            if is_target:
              data,label = self.feature_and_label(color, move, self.go_board, 7)
            # print('color:'+color + '   move:'+str(move))

            self.go_board.apply_move(color, move)

            if is_target:
              yield data, label
          

    
    @classmethod    
    def feature_and_label(cls, color, move, go_board, num_planes):
        '''
        Parameters
        ----------
        color: color of the next person to move
        move: move they decided to make
        go_board: represents the state of the board before they moved

        Planes we write:
        0: our stones with 1 liberty
        1: our stones with 2 liberty
        2: our stones with 3 or more liberties
        3: their stones with 1 liberty
        4: their stones with 2 liberty
        5: their stones with 3 or more liberties
        6: simple ko
        '''

        # print('calling feature and label from seven pane processer')
        row, col = move
        enemy_color = go_board.other_color(color)
        label = row * 19 + col
        move_array = np.zeros((num_planes, go_board.board_size, go_board.board_size))
        for row in range(0, go_board.board_size):
            for col in range(0, go_board.board_size):
                pos = (row, col)
                if go_board.board.get(pos) == color:
                    if go_board.go_strings[pos].liberties.size() == 1:
                        move_array[0, row, col] = 1
                    elif go_board.go_strings[pos].liberties.size() == 2:
                        move_array[1, row, col] = 1
                    elif go_board.go_strings[pos].liberties.size() >= 3:
                        move_array[2, row, col] = 1
                if go_board.board.get(pos) == enemy_color:
                    if go_board.go_strings[pos].liberties.size() == 1:
                        move_array[3, row, col] = 1
                    elif go_board.go_strings[pos].liberties.size() == 2:
                        move_array[4, row, col] = 1
                    elif go_board.go_strings[pos].liberties.size() >= 3:
                        move_array[5, row, col] = 1
                if go_board.is_simple_ko(color, pos):
                    move_array[6, row, col] = 1
        return move_array, label


       
        

