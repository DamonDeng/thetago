import mxnet as mx
import numpy as np
from data_loader.sgf_iter import  SimulatorIter, SGFIter
import logging
from go_core.goboard import GoBoard

class MXNetRobot:
  def __init__(self):
    sym, arg_params, aux_params = mx.model.load_checkpoint('model_zoo/thetago_testing3', 100)
    mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.cpu(0))
    mod.bind(for_training=False, data_shapes=[('data', (1,8,19,19))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    self.model = mod
    self.go_board = GoBoard(19)
    self.board_history_b = np.zeros((8,19,19))
    self.board_history_w = np.zeros((8,19,19))
    self.data_input = np.zeros((1, 8, 19, 19))

  def reset_board(self):
    self.go_board = GoBoard(19)
    self.board_history_b = np.zeros((8,19,19))
    self.board_history_w = np.zeros((8,19,19))
    self.data_input = np.zeros((1, 8, 19, 19))

  def move_window(self, input_array):
    for i in range(len(input_array)-1):
      input_array[i] = input_array[i+1]
    return input_array

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
    if color == 'b':
      self.data_input[0] = self.board_history_w

      data_iter = mx.io.NDArrayIter(self.data_input)

      output = self.model.predict(data_iter)

      position_list = self.get_move(output)

      for position_number in position_list:
        position = self.get_position(position_number)

        if self.go_board.is_move_legal('w', position):
          self.go_board.apply_move('w', position)

          self.board_history_b = self.move_window(self.board_history_b)
          self.board_history_w = self.move_window(self.board_history_w)

          self.board_history_b[-1] = self.go_board.get_state('b')
          self.board_history_w[-1] = self.go_board.get_state('w')
          return position
    else:
      self.data_input[0] = self.board_history_b

      data_iter = mx.io.NDArrayIter(self.data_input)

      output = self.model.predict(data_iter)

      position_list = self.get_move(output)

      for position_number in position_list:
        position = self.get_position(position_number)

        if self.go_board.is_move_legal('b', position):
          self.go_board.apply_move('b', position)
          self.board_history_b = self.move_window(self.board_history_b)
          self.board_history_w = self.move_window(self.board_history_w)

          self.board_history_b[-1] = self.go_board.get_state('b')
          self.board_history_w[-1] = self.go_board.get_state('w')
          return position


  def apply_move(self, color, position):
    if self.go_board.is_move_legal(color, position):
      self.go_board.apply_move(color, position)

      self.board_history_b = self.move_window(self.board_history_b)
      self.board_history_w = self.move_window(self.board_history_w)

      self.board_history_b[-1] = self.go_board.get_state('b')
      self.board_history_w[-1] = self.go_board.get_state('w')

      if color == 'b':
        self.data_input[0] = self.board_history_w

        data_iter = mx.io.NDArrayIter(self.data_input)

        output = self.model.predict(data_iter)

        position_list = self.get_move(output)

        for position_number in position_list:
          position = self.get_position(position_number)

          if self.go_board.is_move_legal('w', position):
            self.go_board.apply_move('w', position)

            self.board_history_b = self.move_window(self.board_history_b)
            self.board_history_w = self.move_window(self.board_history_w)

            self.board_history_b[-1] = self.go_board.get_state('b')
            self.board_history_w[-1] = self.go_board.get_state('w')
            return True, position
      else:
        self.data_input[0] = self.board_history_b

        data_iter = mx.io.NDArrayIter(self.data_input)

        output = self.model.predict(data_iter)

        position_list = self.get_move(output)

        for position_number in position_list:
          position = self.get_position(position_number)

          if self.go_board.is_move_legal('b', position):
            self.go_board.apply_move('b', position)
            self.board_history_b = self.move_window(self.board_history_b)
            self.board_history_w = self.move_window(self.board_history_w)

            self.board_history_b[-1] = self.go_board.get_state('b')
            self.board_history_w[-1] = self.go_board.get_state('w')
            return True, position
    else:
      return False, None
      

