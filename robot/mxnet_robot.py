import mxnet as mx
import numpy as np
from data_loader.sgf_iter import  SimulatorIter, SGFIter
import logging
from go_core.goboard import GoBoard
from go_core.array_goboard import ArrayGoBoard

import copy
from data_loader.original_processor import OriginalProcessor
from data_loader.value_processor import ValueProcessor
from robot.mc_tree_searcher import MCTreeSearcher
from go_core.space_manager import SpaceManager

class MXNetRobot:
  def __init__(self, checkpoint_file, epoch, processor_class, value_file=None, value_epoch=None, value_processor_class = ValueProcessor):
    self.board_size = 19

    sym, arg_params, aux_params = mx.model.load_checkpoint(checkpoint_file, epoch)
    mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.cpu(0))
    data_shapes = processor_class.get_data_shape(1)
    mod.bind(for_training=False, data_shapes=data_shapes, label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    self.model = mod

    self.value_model = None
    if value_file is not None and value_epoch is not None:
      value_sym, value_arg_params, value_aux_params = mx.model.load_checkpoint(value_file, value_epoch)
      value_mod = mx.mod.Module(symbol=value_sym, label_names=None, context=mx.cpu(0))
      value_mod.bind(for_training=False, data_shapes=data_shapes, label_shapes=value_mod._label_shapes)
      value_mod.set_params(value_arg_params, value_aux_params, allow_missing=True)
      self.value_model = value_mod
    
    self.go_board = ArrayGoBoard(self.board_size, eye_checking=True)
    # self.space_manager = SpaceManager(19)
    # print (self.space_manager)
    self.processor_class = processor_class
    self.value_processor_class = value_processor_class
    

  def get_board(self):
    return self.go_board

  def reset_board(self):
    self.go_board = ArrayGoBoard(self.board_size, eye_checking=True)

  def get_position(self, input_number):
    if input_number == 361:
      return None
    else:
      row = int(input_number/self.board_size)
      col = input_number%self.board_size

      return (row, col)

  def get_move(self, predict_result):
    # sort the predict result and return
    output_numpy = predict_result.asnumpy()

    output_numpy = np.squeeze(output_numpy)
   
    position_list = np.argsort(output_numpy)[::-1]    

    return position_list

  def select_move(self, color):

    # we should modify this part of code to make it auto
    data,label = self.processor_class.feature_and_label(color, (0,0), self.go_board)

    (input_data_label, input_data_shape) = self.processor_class.get_single_data_shape()[0]

    input_data = np.zeros(input_data_shape)
    
    input_data[0] = data

    data_iter = mx.io.NDArrayIter(input_data)

    
    output = self.model.predict(data_iter)
    output_np = output.asnumpy()[0]

    # print("#" + str(output.asnumpy()))

    position_list = self.get_move(output)

    # print("#" + str(position_list))

    # get first 10 position
    selected_position_list = []
    selected_value_list = []
    max_selected_number = 2
    selected_number = 0
    for position_number in position_list:
      position = self.get_position(position_number)
      if self.go_board.is_move_legal(color, position):
        if not self.go_board.is_my_eye(color, position):
          selected_position_list.append(position)
          selected_value_list.append(output_np[position_number])
          selected_number = selected_number + 1
          if selected_number >= max_selected_number:
            break

    if self.value_model is None:
      if len(selected_position_list) < 1:
        return None
      else:

        # temp_board = copy.deepcopy(self.go_board)
        # tree_searcher = MCTreeSearcher(temp_board, self.model, self.processor_class)

        # print('## trying to search')
        # tree_searcher.search(color)

        self.go_board.apply_move(color, selected_position_list[0])
        # self.space_manager.apply_move(color, selected_position_list[0])
        # print (self.space_manager)

        # print("# possible moves:"+str(selected_position_list))
        # print("# move value:" + str(selected_value_list))
        return selected_position_list[0]
    else:
      ## should evaluate the position value here
      if len(selected_position_list) < 1:
        return None
      else:
        print("## possible moves:"+str(selected_position_list))
        print("## move value:" + str(selected_value_list))
        
        value_input_data = np.zeros((1,7,19,19))  
        result_list = []
        max_value = -1
        for cur_selected_position in selected_position_list:
          # print("## in the selected loop")
          temp_board = copy.deepcopy(self.go_board)
          temp_board.apply_move(color, cur_selected_position)
          value_data, value_label = self.value_processor_class.feature_and_label(color, None, temp_board)
          value_input_data[0] = value_data
          
          value_data_iter = mx.io.NDArrayIter(value_input_data)
          value_output = self.value_model.predict(value_data_iter).asnumpy()
          # print("## value_output of " + str(cur_selected_position) + " is:" + str(value_output[0]))
          # print('## max value is:' + str(max_value))
          if value_output[0] > max_value:
            max_value = value_output[0]
            result_position = cur_selected_position
          
        print ("#result max is:"+str(max_value))
        print ("#result position is: "+str(result_position))


        self.go_board.apply_move(color, result_position)
        # self.space_manager.apply_move(color, result_position)
        # print (self.space_manager)

        return result_position

    


   


  def apply_move(self, color, move):
    # print ("# applying move: " + color + " " + str(move))
    self.go_board.apply_move(color, move)
    # self.space_manager.apply_move(color, move)
    # print (self.go_board)

    if self.value_model is not None:
      # trying to compute the evaluation value of current board
      value_input_data = np.zeros((1,7,19,19))  

      value_data, value_label = self.value_processor_class.feature_and_label(color, None, self.go_board)
      value_input_data[0] = value_data
      
      value_data_iter = mx.io.NDArrayIter(value_input_data)
      value_output = self.value_model.predict(value_data_iter).asnumpy()

      print ("# after applying move, value of "+color+" is:" + str(value_output[0]))

      enemy_color = self.go_board.other_color(color)

      value_data, value_label = self.value_processor_class.feature_and_label(enemy_color, None, self.go_board)
      value_input_data[0] = value_data
      
      value_data_iter = mx.io.NDArrayIter(value_input_data)
      value_output = self.value_model.predict(value_data_iter).asnumpy()

      print ("# after applying move, value of "+enemy_color+" is:" + str(value_output[0]))


      
  def get_score(self):
    return self.go_board.get_score()

  def analyst_result(self):
    self.go_board.analyst_result()
