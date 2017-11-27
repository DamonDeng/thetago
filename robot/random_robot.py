import mxnet as mx
import numpy as np
from data_loader.sgf_iter import  SimulatorIter, SGFIter
import logging
from go_core.goboard import GoBoard
import copy
from data_loader.original_processor import OriginalProcessor
from data_loader.value_processor import ValueProcessor
from robot.mc_tree_searcher import MCTreeSearcher

class RadomRobot:
  def __init__(self):
    self.go_board = GoBoard(19)
    
  def set_board(self, board):
    self.go_board = copy.deepcopy(board)

  def reset_board(self):
    self.go_board = GoBoard(19)

  def get_position(self, input_number):
    row = int(input_number/19)
    col = input_number%19
    return (row, col)


  def select_move(self, color):

  
      if self.go_board.is_move_legal(color, position):
        

    

    


   


  def apply_move(self, color, move):
    self.go_board.apply_move(color, move)

   


      

