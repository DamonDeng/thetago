import mxnet as mx
import numpy as np
import random
import os

from gosgf import Sgf_game
from go_core.goboard import GoBoard
from data_loader.data_generator import DataGenerator

class SGFIter(mx.io.DataIter):
    def __init__(self, sgf_directory, batch_size=30, file_limit = -1):
        self.sgf_directory = sgf_directory
        self.batch_size = batch_size
        self.cur_batch = 0
        self.board_col = 19
        self.board_row = 19
        self.history_length = 8
        self.board_length = self.board_col * self.board_row
        self.file_limit = file_limit
        self.data_generator = None        

        print ('initing SGFIter...')
        self.file_list = []
        for file in os.listdir(self.sgf_directory):  
          file_path = os.path.join(self.sgf_directory, file)  
          if os.path.splitext(file_path)[1]=='.sgf':  
              self.file_list.append(file_path) 

        self.cur_file_index = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_file_index = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return [('data',(self.batch_size, self.history_length, self.board_row, self.board_col))]
        

    @property
    def provide_label(self):
        return [('softmax_label', (self.batch_size, ))]
        

    def next(self):
      success, batch_data, batch_label = self.get_current_batch()
      if success:
        data = [mx.nd.array(batch_data)]
        label = [mx.nd.array(batch_label)]
        return mx.io.DataBatch(data, label)
      else:
        raise StopIteration

    def get_current_batch(self):  

      batch_data = np.zeros((self.batch_size, self.history_length, self.board_row, self.board_col))
      batch_label = np.zeros((self.batch_size,))

      if self.data_generator is None:
        if self.cur_file_index >= len(self.file_list) or self.cur_file_index > self.file_limit:
            return False, None, None
        else:
          cur_file_name = self.file_list[self.cur_file_index]
          self.cur_file_index = self.cur_file_index + 1
          self.data_generator = DataGenerator(cur_file_name)

      for i in range(self.batch_size):
        
        get_next = False
        while (not get_next):
          try:
            data, label = self.data_generator.next() 
            get_next = True
            batch_data[i] = data
            # print(label)
            # print(batch_label)
            batch_label[i] = label
                  
          except StopIteration:
            if self.cur_file_index >= len(self.file_list) or self.cur_file_index > self.file_limit:
              return False, None, None
            else:
              cur_file_name = self.file_list[self.cur_file_index]
              self.cur_file_index = self.cur_file_index + 1
              self.data_generator = DataGenerator(cur_file_name)
      
      return True, batch_data, batch_label
            

class SimpleIter(mx.io.DataIter):
    def __init__(self, data_names, data_shapes, data_gen,
                 label_names, label_shapes, label_gen, num_batches=10):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            for d,g in zip(self._provide_data, self.data_gen):
              print ("d:" + str(d))
              print ("g:" + str(g))
              print ("g(d[1]:" + str(g(d[1])))
            data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            return mx.io.DataBatch(data, label)
        else:
            raise StopIteration

class SimulatorIter(mx.io.DataIter):
    def __init__(self, batch_size=30, num_batches=10):
        self.num_batches = num_batches
        self.cur_batch = 0
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return [('data',(self.batch_size ,8,19,19))]
        # return zip(['data'],[(32,100)])

    @property
    def provide_label(self):
        return [('softmax_label', (self.batch_size ,))]
        # return zip(['softmax_label'],[(32,)])

    def next(self):
        if self.cur_batch < self.num_batches:
            print("Return: " + str(self.cur_batch))
            self.cur_batch += 1
            data_pool = np.random.uniform(-1, 1, (self.batch_size , 8, 19, 19))
            label_pool = np.random.randint(0, 361, (self.batch_size ,))

            data = [mx.nd.array(data_pool)]
            label = [mx.nd.array(label_pool)]
            return mx.io.DataBatch(data, label)
        else:
            raise StopIteration