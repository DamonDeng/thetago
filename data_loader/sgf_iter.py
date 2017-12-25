import mxnet as mx
import numpy as np
import random
import os
import multiprocessing
import signal
import six.moves.queue as queue
import time
import tarfile

from gosgf import Sgf_game
from go_core.goboard import GoBoard
from data_loader.feature_processor import FeatureProcessor



class SGFIter(mx.io.DataIter):
    def __init__(self, sgf_directory, batch_size=30, file_limit = -1, processor_class=FeatureProcessor):
        self.sgf_directory = sgf_directory
        self.batch_size = batch_size
        
        self.board_col = 19
        self.board_row = 19

        self.processor_class = processor_class

        self.board_length = self.board_col * self.board_row
        self.file_limit = file_limit



        self.batch_data = np.zeros(self.processor_class.get_data_shape_only(batch_size))
        self.batch_label = np.zeros(self.processor_class.get_label_shape_only(batch_size))
            

        print ('initing SGFIter...')
        self.file_list = []
        number_of_file = 0
        for file in os.listdir(self.sgf_directory):  
          if self.file_limit > 0 and number_of_file > self.file_limit:
              break  
          number_of_file+=1
          file_path = os.path.join(self.sgf_directory, file)  
          if os.path.splitext(file_path)[1]=='.tar':  
              self.file_list.append(file_path) 

        self.generator = self.get_generator()

      

    def __iter__(self):
        return self

    def reset(self):
        self.generator = self.get_generator()

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self.processor_class.get_data_shape(self.batch_size)

        
        

    @property
    def provide_label(self):
        return self.processor_class.get_label_shape(self.batch_size)
        

    def next(self):

    #   print('got one batch')
      try:
          
          for i in range(self.batch_size):
              data, label = self.generator.next()
              
              self.batch_data[i] = data
              self.batch_label[i] = label
          
          data = [mx.nd.array(self.batch_data)]
          label = [mx.nd.array(self.batch_label)]
      except StopIteration:
          raise StopIteration

      return mx.io.DataBatch(data, label)

        


    def get_generator(self):
        for file_name in self.file_list:

            this_zip = tarfile.open(file_name)
            name_list = this_zip.getnames()
            for name in name_list:
                if name.endswith('.sgf'):
                    sgf_content = this_zip.extractfile(name).read()

                    processor = self.processor_class.get_processor(sgf_content)

                    features = processor.get_generator()

                    for feature in features:

                        data, label = feature
                        yield data, label
                
   

class SimulatorIter(mx.io.DataIter):
    def __init__(self, batch_size=30, history_length=8, num_batches=10):
        self.num_batches = num_batches
        self.cur_batch = 0
        self.history_length = history_length
        self.batch_size = batch_size
        self.data_pool = np.random.uniform(-1, 1, (self.batch_size , self.history_length, 19, 19))
        self.label_pool = np.random.randint(0, 361, (self.batch_size ,))


    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return [('data',(self.batch_size, self.history_length, 19, 19))]
        # return zip(['data'],[(32,100)])

    @property
    def provide_label(self):
        return [('softmax_label', (self.batch_size ,))]
        # return zip(['softmax_label'],[(32,)])

    def next(self):
        if self.cur_batch < self.num_batches:
            # print("Return: " + str(self.cur_batch))
            self.cur_batch += 1

            data = [mx.nd.array(self.data_pool)]
            label = [mx.nd.array(self.label_pool)]
            return mx.io.DataBatch(data, label)
        else:
            raise StopIteration