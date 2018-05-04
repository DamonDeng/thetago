from gosgf import Sgf_game
from go_core.array_goboard import ArrayGoBoard

import numpy as np
import mxnet as mx

class ZeroDualResProcessor(object):

    def __init__(self, sgf_content, board_size=19, level_limit='0d', player='all'):
        # self.sgf_file = sgf_file

        # print ('initing the Zero Dual Res processor!')
        self.board_col = board_size
        self.board_row = board_size
        self.level_limit = level_limit # the value should be nk, nd, np, n is the level number
        self.player = player # the value should be 'all', 'winner' or 'loser'
        self.board_length = self.board_row * self.board_col

        self.go_board = ArrayGoBoard(self.board_col)
        
        self.label = 0 # np.zeros((self.board_row * self.board_col))

        # sgf_content = open(sgf_file,'r').read()
      
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
      return (batch_size, 17, board_row, board_col)

    @classmethod
    def get_label_shape_only(cls, batch_size):
      return [(batch_size, ), (batch_size, )]

    @classmethod
    def get_data_shape(cls, batch_size):
      board_col = 19
      board_row = 19
      return [('data',(batch_size, 17, board_row, board_col))]

    @classmethod
    def get_single_data_shape(cls):
      board_col = 19
      board_row = 19
      return [('data',(1, 17, board_row, board_col))]

    @classmethod
    def get_label_shape(cls, batch_size):
      return [('move_label', (batch_size, )), ('value_label', (batch_size, ))]


    def get_generator(self):

      # using new function is_level_heighter_than to filter games play by player higher than limit level
      if self.sgf.is_level_higher_than(self.level_limit):

        winner = self.sgf.get_winner()
        handicap = self.sgf.get_handicap()

        # print('handicap'),
        # print(handicap)

        if handicap is not None:
          #only process the noral game
          return

        result = self.sgf.get_winner_result()

        # if result is not None:
        #   print ('result is:' + result)

        if result is None:
          return
        else:
          if not isinstance(result, float):
            if not result == 'resign':
              return
          # else:
          #   print('result:'),
          #   print(result)
          

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
            
          
          if not color is None:   # and not move is None: # move is None, it is a pass move

            # print('-------------------------------')
            if is_target:
              data,move_label,value_label = self.feature_and_label(color, move, winner, self.go_board)
            # print('color:'+color + '   move:'+str(move))

            if not move is None:
              self.go_board.apply_move(color, move)

            if is_target:
              yield data, move_label, value_label
        

    
    @classmethod    
    def feature_and_label(cls, color, move, winner, go_board):
        '''
        Parameters
        ----------
        color: color of the next person to move
        move: move they decided to make
        go_board: represents the state of the board before they moved

        Planes we write:
        0~7: current stone history
        8~15: enemy stone history
        16: current color

        '''

        # print('calling feature and label from seven pane processer')

        history_length = 8

        # history_length = 2

        if move == None:
          move_label = 361
        else:
          row, col = move
          move_label = row * 19 + col
        
        move_array = go_board.get_move_array(history_length, color)

        # print ('winner is: ' + winner)
        # print ('color is:' + color)

        if winner is None:
          value_label = 0
        else:
          if color == winner:
            value_label = 1
          else:
            value_label = -1

        return move_array, move_label, value_label


       
        

