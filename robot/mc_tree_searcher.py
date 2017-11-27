import mxnet as mx
import numpy as np
from data_loader.sgf_iter import  SimulatorIter, SGFIter
import logging
from go_core.goboard import GoBoard
import copy
from data_loader.original_processor import OriginalProcessor
from data_loader.value_processor import ValueProcessor
import time

class MCTreeSearcher:
  def __init__(self, go_board, policy_model, policy_processor_class, value_model = None, value_processor_class = None):
    self.go_board = go_board
    self.policy_model = policy_model
    self.policy_processor_class = policy_processor_class
    self.value_model = value_model
    self.value_processor_class = value_processor_class

  def search(self, color):

    print('## in the search method')
    wait_start_ts = time.time()
    
    policy_search_time = 50

    for i in range(policy_search_time):
      data,label = self.policy_processor_class.feature_and_label(color, (0,0), self.go_board, 7)

      input_data = np.zeros((1,7,19,19))
      
      input_data[0] = data

      data_iter = mx.io.NDArrayIter(input_data)

      
      output = self.policy_model.predict(data_iter)
      output_np = output.asnumpy()[0]

      wait_end_ts = time.time()

    print("## time used: %.5f seconds" % ((wait_end_ts - wait_start_ts)/policy_search_time,))


   