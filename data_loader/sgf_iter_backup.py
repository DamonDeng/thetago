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
from data_loader.feature_generator import FeatureGenerator

from betago.corpora import build_index, find_sgfs, load_index, store_index

from betago.processor import SevenPlaneProcessor


class DumyIter(mx.io.DataIter):
    def __init__(self,  batch_size=30, history_length=8):
        
        self.batch_size = batch_size
        self.cur_batch = 0
        self.board_col = 19
        self.board_row = 19
        self.history_length = history_length
           

        print ('initing DumyIter...')
        

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
        batch_data = np.zeros((self.batch_size, self.history_length, self.board_row, self.board_col))
        batch_label = np.zeros((self.batch_size,))

        data = [mx.nd.array(batch_data)]
        label = [mx.nd.array(batch_label)]
        
        return mx.io.DataBatch(data, label)
      


class SGFIter(mx.io.DataIter):
    def __init__(self, sgf_directory, batch_size=30, file_limit = -1, history_length=8):
        self.sgf_directory = sgf_directory
        self.batch_size = batch_size
        self.cur_batch = 0
        self.board_col = 19
        self.board_row = 19
        self.history_length = history_length
        self.board_length = self.board_col * self.board_row
        self.file_limit = file_limit
        self.feature_generator = None        

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

      if self.feature_generator is None:
        if self.cur_file_index >= len(self.file_list) or self.cur_file_index > self.file_limit:
            return False, None, None
        else:
          cur_file_name = self.file_list[self.cur_file_index]
          self.cur_file_index = self.cur_file_index + 1
          self.feature_generator = FeatureGenerator(cur_file_name, history_length=self.history_length)

      for i in range(self.batch_size):
        
        get_next = False
        while (not get_next):
          try:
            data, label = self.feature_generator.next() 
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
              self.feature_generator = FeatureGenerator(cur_file_name, history_length=self.history_length)
      
      return True, batch_data, batch_label

class SGFMultiThreadIter(mx.io.DataIter):
    def __init__(self, index_file):
        self.index_file = index_file
        self.corpus_index = load_index(open(index_file))

    def testing(self):
        print ('start testing')

        print ('chunk_size is: '+ str(self.corpus_index.chunk_size))

        # print(self.corpus_index.physical_files)

        # for pointer in self.corpus_index.boundaries:
        #     print (pointer)

        workers = 1

        q = multiprocessing.Queue(workers)
        stop_q = multiprocessing.Queue()
        p = multiprocessing.Process(target=prepare_training_data,
                                    args=(workers, 3, self.corpus_index, q, stop_q))
        
        p.start()

        try:
            while True:
                print("Waiting for prepared training chunk...")
                wait_start_ts = time.time()
                X,Y = q.get()
                wait_end_ts = time.time()
                print("Idle %.1f seconds" % (wait_end_ts - wait_start_ts,))
                
        finally:
            print("Shutting down workers, please wait...")
            stop_q.put(1)
            stop_q.close()


def _disable_keyboard_interrupt():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def _prepare_training_data_single_process(worker_idx, chunk, corpus_index, output_q, stop_q):
    
    _disable_keyboard_interrupt()
    processor = SevenPlaneProcessor()

    chunk = corpus_index.get_chunk(chunk)
    xs, ys = [], []
    for board, next_color, next_move in chunk:
        if not stop_q.empty():
            print("Got stop signal, aborting.")
            return
        feature, label = processor.feature_and_label(next_color, next_move, board,
                                                     processor.num_planes)
        xs.append(feature)
        ys.append(label)
    X = np.array(xs)
    Y = np.array(ys)

    # for i in range(1000):
    #     for j in range(1000):
    #         # for k in range(10):
    #         X = [[1,2,3], [4,5,6]]
    #         Y = [[1,2,3], [4,5,6]]
    
    print('in single thread, thread id:' + str(worker_idx))         
    output_q.put((worker_idx, X, Y))
    output_q.close()

def prepare_training_data(num_workers, next_chunk, corpus_index, output_q, stop_q):
    _disable_keyboard_interrupt()
    stopped = False

    while True:
        if not stop_q.empty():
            output_q.close()
            output_q.join_thread()
            stop_q.close()
            stop_q.join_thread()
            return
        print ('in prepare_training_data thread')
        chunks_to_process = []
        for _ in range(num_workers):
            chunks_to_process.append(next_chunk)
            next_chunk = (next_chunk + 1) % corpus_index.num_chunks
        workers = []
        inter_q = multiprocessing.Queue()
        for i, chunk in enumerate(chunks_to_process):
            workers.append(multiprocessing.Process(
                target=_prepare_training_data_single_process,
                args=(i, chunk, corpus_index, inter_q, stop_q)))
        for worker in workers:
            worker.start()
        results = []
        while len(results) < len(workers):
            try:
                results.append(inter_q.get(block=True, timeout=1))
            except queue.Empty:
                if not stop_q.empty():
                    stopped = True
                    break
        for worker in workers:
            worker.join()
        inter_q.close()
        inter_q.join_thread()
        if stopped:
            output_q.close()
            return
        assert len(results) == len(workers)
        results.sort()
        for _, X, Y in results:
            output_q.put((X, Y))

class SGFDirIter(mx.io.DataIter):
    def __init__(self, sgf_directory, file_iter, file_limit = -1):
        self.sgf_directory = sgf_directory
        self.file_iter = file_iter
        self.file_limit = file_limit
        
        print ('Initing SGFDirIter with directory:')
        print (self.sgf_directory)

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
        return self.file_iter.provide_data
        

    @property
    def provide_label(self):
        return self.file_iter.provide_label
        

    def next(self):
      if not self.file_iter.hasFile():
        success = self.set_next_file()
        if not success:
          raise StopIteration
      
      
      try:
        data_batch = self.file_iter.next()
        return data_batch
      except StopIteration:
        success = self.set_next_file()
          if not success:
            raise StopIteration
        

      success, batch_data, batch_label = self.get_current_batch()
      if success:
        data = [mx.nd.array(batch_data)]
        label = [mx.nd.array(batch_label)]
        return mx.io.DataBatch(data, label)
      else:
        raise StopIteration

    def set_next_file():
      if self.cur_file_index >= len(self.file_list) or self.cur_file_index > self.file_limit:
        return False
      else:
        cur_file_name = self.file_list[self.cur_file_index]
        self.cur_file_index = self.cur_file_index + 1
        self.fileIter.set_file(cur_file_name)
        return True


    def get_current_batch(self):  

      batch_data = np.zeros((self.batch_size, self.history_length, self.board_row, self.board_col))
      batch_label = np.zeros((self.batch_size,))

      if self.feature_generator is None:
        if self.cur_file_index >= len(self.file_list) or self.cur_file_index > self.file_limit:
            return False, None, None
        else:
          cur_file_name = self.file_list[self.cur_file_index]
          self.cur_file_index = self.cur_file_index + 1
          self.feature_generator = FeatureGenerator(cur_file_name, history_length=self.history_length)

      for i in range(self.batch_size):
        
        get_next = False
        while (not get_next):
          try:
            data, label = self.feature_generator.next() 
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
              self.feature_generator = FeatureGenerator(cur_file_name, history_length=self.history_length)
      
      return True, batch_data, batch_label
            

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