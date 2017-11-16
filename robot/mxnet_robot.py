import mxnet as mx
import numpy as np
from data_loader.sgf_iter import  SimulatorIter, SGFIter
import logging
from go_core.goboard import GoBoard
import copy
from data_loader.original_processor import OriginalProcessor

class MXNetRobot:
  def __init__(self, checkpoint_file, epoch, processor_class):
    sym, arg_params, aux_params = mx.model.load_checkpoint(checkpoint_file, epoch)
    mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.cpu(0))
    mod.bind(for_training=False, data_shapes=[('data', (1,7,19,19))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    self.model = mod
    self.go_board = GoBoard(19)
    self.processor_class = processor_class
    
  def set_board(self, board):
    self.go_board = copy.deepcopy(board)

  def reset_board(self):
    self.go_board = GoBoard(19)

  def get_position(self, input_number):
    row = int(input_number/19)
    col = input_number%19

    return (row, col)

  def get_move(self, predict_result):
    output_numpy = predict_result.asnumpy()

    output_numpy = np.squeeze(output_numpy)
   
    position_list = np.argsort(output_numpy)[::-1]    

    return position_list

  def select_move(self, color):

    data,label = self.processor_class.feature_and_label(color, (0,0), self.go_board, 7)

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

    input_data = np.zeros((1,7,19,19))
    
    input_data[0] = data

    data_iter = mx.io.NDArrayIter(input_data)

    
    output = self.model.predict(data_iter)

    position_list = self.get_move(output)

    for position_number in position_list:
      position = self.get_position(position_number)

      if self.go_board.is_move_legal(color, position):
        self.go_board.apply_move(color, position)
        return position

    


   


  def apply_move(self, color, move):
    self.go_board.apply_move(color, move)
      

