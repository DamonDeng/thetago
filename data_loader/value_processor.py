from gosgf import Sgf_game
from go_core.goboard import GoBoard

import numpy as np
import mxnet as mx

class ValueProcessor(object):

    def __init__(self, sgf_file, board_size=19):
        self.sgf_file = sgf_file
        self.board_col = board_size
        self.board_row = board_size
        
        self.board_length = self.board_row * self.board_col

        self.go_board = GoBoard(self.board_col)
        
        self.label = 0 # np.zeros((self.board_row * self.board_col))

        sgf_content = open(sgf_file,'r').read()
      
        sgf = Sgf_game.from_string(sgf_content)

        self.main_sequence_iter = sgf.main_sequence_iter()

        self.winner = sgf.get_winner()
        
        if sgf.get_handicap() != None and sgf.get_handicap() != 0:
          # print('handling handicap')
          for setup in sgf.get_root().get_setup_stones():
            for move in setup:
              self.go_board.apply_move('b', move)


    @classmethod
    def get_processor(cls, sgf_file):
      return cls(sgf_file)

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
      return [('label', (batch_size, ))]
      # return [('softmax_label', (batch_size, ))]


    def get_generator(self):
      
      if self.winner is not None:
        for item in self.main_sequence_iter:

          color, move = item.get_move()

          if not color is None and not move is None:

            # as we are evaluating the vaule of current move, we should apply the move at first
            self.go_board.apply_move(color, move)
            # print('-------------------------------')
            data,label = self.feature_and_label(color, self.winner, self.go_board)
            # print('color:'+color + '   move:'+str(move))
            
            # panenumber = 0
            # for pane in data:
            #   rownumber = 0
            #   for row in pane:
            #     columnnumber = 0
            #     for column in row:
            #       if column != 0:
            #         print("("+str(panenumber)+","+str(columnnumber)+","+str(rownumber)+"):" + str(column)),
            #       columnnumber = columnnumber + 1 
            #     rownumber = rownumber + 1
            #   panenumber = panenumber + 1
            
            # print(" ")
            # print(label)

            

            yield data, label
          

    
    @classmethod    
    def feature_and_label(cls, color, winner, go_board):
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
        num_planes = 7

        enemy_color = go_board.other_color(color)
        if winner is None:
            label = None
        else:
            if winner == color:
                # print("winner is:"+winner+"  color is:"+color +"   label is 1")
                # print("label is 1")
                label = 1
            else:
                # print("winner is:"+winner+"  color is:"+color+"   label is 0")
                # print("label is 0")
                label = -1
        
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


       
        

