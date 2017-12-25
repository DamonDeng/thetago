import mxnet as mx
import numpy as np
import random
import os
import multiprocessing
import signal
import six.moves.queue as queue
import time

from gosgf import Sgf_game
from go_core.goboard import GoBoard
from data_loader.feature_processor import FeatureProcessor
from data_loader.original_processor import OriginalProcessor

import argparse
import importlib
import multiprocessing
import os
import signal
import sys
import time
import tarfile

import numpy as np
import six.moves.queue as queue



class NDArrayConverter(object):
    def __init__(self, sgf_directory, workers=1, batch_size=30, file_limit = -1, processor_class=FeatureProcessor, level_limit='0d', player='all'):
        self.sgf_directory = sgf_directory
        self.workers=workers
        self.batch_size = batch_size
        self.level_limit = level_limit
        self.player = player
        
        self.board_col = 19
        self.board_row = 19

        self.processor_class = processor_class

        self.board_length = self.board_col * self.board_row
        self.file_limit = file_limit



        self.batch_data = np.zeros(self.processor_class.get_data_shape_only(batch_size))
        self.batch_label = np.zeros(self.processor_class.get_label_shape_only(batch_size))
            

        print ('initing multiple thread SGFIter...')
        self.file_list = []
        for i in range(workers):
          self.file_list.append([])

        number_of_file = 0
        worker_number = 0

        for file in os.listdir(self.sgf_directory):  
          if self.file_limit > 0 and number_of_file > self.file_limit:
              break  
          number_of_file+=1
          file_path = os.path.join(self.sgf_directory, file)  
          if os.path.splitext(file_path)[1]=='.tar':  
              worker_index = worker_number%self.workers
              worker_number = worker_number + 1
              self.file_list[worker_index].append(file_path) 
              print('tar file found:' + file_path)

        print ("need to process: " + str(number_of_file-1) + " files")
        self.q = multiprocessing.Queue(maxsize=2 * self.workers)
        self.stop_q = multiprocessing.Queue()
        self.p = multiprocessing.Process(target=prepare_training_data,
                                    args=(self.workers, self.file_list, self.processor_class, self.q, self.stop_q, self.level_limit, self.player))
        self.p.start()

        # self.generator = self.get_generator()


    
      

    def __iter__(self):
        return self

    def reset(self):
      self.stop_task()
      # print('task stoped, waiting for 10 seconds')
      # time.sleep(10)

      self.q = multiprocessing.Queue(maxsize=2 * self.workers)
      self.stop_q = multiprocessing.Queue()
      self.p = multiprocessing.Process(target=prepare_training_data,
                                  args=(self.workers, self.file_list, self.processor_class, self.q, self.stop_q, self.level_limit, self.player))
      self.p.start()
      # print('after trying to run the thread')

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self.processor_class.get_data_shape(self.batch_size)
     

    @property
    def provide_label(self):
        return self.processor_class.get_label_shape(self.batch_size)
        

    def next(self):
        
        # print('calling next function to get data')
        for i in range(self.batch_size):
            try:
              data, label = self.q.get(block=True, timeout=1)
            except queue.Empty:
              if not self.stop_q.empty():
                raise StopIteration
            self.batch_data[i] = data
            self.batch_label[i] = label
        # print('got one batch of data')
        data = [mx.nd.array(self.batch_data)]
        label = [mx.nd.array(self.batch_label)]
        

        return mx.io.DataBatch(data, label)

    def stop_task(self):
      # print('trying to drain the output q')
      while not self.q.empty():
        self.q.get()
      self.q.close()
      self.q.join_thread()
      # print("Shutting down workers, please wait...")
      # self.stop_q.put(1)
      self.stop_q.close()
      self.p.join()
      # print('after stop queue close')

                
   
def prepare_training_data(workers, file_list, processor_class, output_q, stop_q, level_limit='0d', player='all'):
  _disable_keyboard_interrupt()
  workers = []
  
  for inter_file_list in file_list:
      workers.append(multiprocessing.Process(
          target=_prepare_training_data_single_process,
          args=(inter_file_list, processor_class, output_q, stop_q, level_limit, player)))

  for worker in workers:
      worker.start()

  for worker in workers:
    worker.join()

  stop_q.put(1)


def _prepare_training_data_single_process(inter_file_list, processor_class, output_q, stop_q, level_limit='0d', player='all'):
  # Make sure ^C gets handled in the main process.
  _disable_keyboard_interrupt()

  for file_name in inter_file_list:

    this_zip = tarfile.open(file_name)
    name_list = this_zip.getnames()
    for name in name_list:
        if name.endswith('.sgf'):
            sgf_content = this_zip.extractfile(name).read()

            processor = processor_class.get_processor(sgf_content, level_limit=level_limit, player=player)

            features = processor.get_generator()

            for feature in features:

                data, label = feature
                output_q.put((data, label))
                if not stop_q.empty():
                    print("Got stop signal, aborting.")
                    return
  

def _disable_keyboard_interrupt():
    signal.signal(signal.SIGINT, signal.SIG_IGN)  

