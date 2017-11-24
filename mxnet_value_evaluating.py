import mxnet as mx
import numpy as np
from data_loader.sgf_iter import SGFIter, SimulatorIter
from data_loader.multi_thread_sgf_iter import MultiThreadSGFIter
import logging
from data_loader.feature_processor import FeatureProcessor
from data_loader.original_processor import OriginalProcessor
from data_loader.value_processor import ValueProcessor

from sys import argv
import sys
import importlib
import os
import argparse
from multiprocessing import cpu_count

def start_eval(args):

  print('mxnet value evaluating starting 0.03')

  

  logging.basicConfig(level=logging.INFO)
  
  # Need automaticlly way to load the class
  if args.processor == 'FeatureProcessor':
    processor = FeatureProcessor
  elif args.processor == 'ValueProcessor':
    processor = ValueProcessor
  elif args.processor == 'OriginalProcessor':
    processor = OriginalProcessor
  else:
    processor = OriginalProcessor

  if args.workers == -1:  
    workers = cpu_count()
  else:
    workers = args.workers

  value_sym, value_arg_params, value_aux_params = mx.model.load_checkpoint(args.valuefile, args.valueepoch)
  value_mod = mx.mod.Module(symbol=value_sym, label_names=None, context=mx.cpu(0))
  value_mod.bind(for_training=False, data_shapes=[('data', (1,7,19,19))], label_shapes=value_mod._label_shapes)
  value_mod.set_params(value_arg_params, value_aux_params, allow_missing=True)
  

  print('using '+ str(workers) + " workers to load the SGF data")
  data_iter = MultiThreadSGFIter(sgf_directory=args.data, workers=workers, batch_size=args.batchsize, file_limit = args.filelimit, processor_class=processor)
  #data_iter = SimulatorIter( batch_size=1024)
  #data_iter = SimulatorIter(batch_size=1024, num_batches=1024)
 
  print (str(data_iter.provide_data))
  print (str(data_iter.provide_label))
  
  
  # try:
  # score = value_mod.score(data_iter, ['mse'], 
  #                         batch_end_callback=mx.callback.Speedometer(32, 20))
  # print("Accuracy score is %f" % (score[0][1]))

  print('trying to get the data and label')
 
  
  total_number = 0
  right_number = 0

  for i in range(300):
    data_label = data_iter.next()
    eval_label = data_label.label[0].asnumpy()
    output_label = value_mod.predict(mx.io.NDArrayIter(data_label.data[0])).asnumpy()

    for cur_label in zip(eval_label, output_label):
      eval_label_value, output_label_value = cur_label
      if output_label_value[0] > 0.5:
        output_label_result = 1
      else:
        output_label_result = 0

      total_number = total_number + 1
      if eval_label_value == output_label_result:
        right_number = right_number + 1

    print("Accuracy: " + str(right_number) + "/" + str(total_number) + "=" + str(float(right_number)/total_number))
    # print ("right:output    " + str(eval_label_value) + ":" +str(output_label_result))

  # print('result: ' + str(output_label.asnumpy))

  # print('and the label' + label.asnumpy())

  # output_label = value_mod.predict(data)

  # print(output_label.asnumpy())

  # finally:
  #   data_iter.stop_task()




def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('eval', help='Do some training.')
    train_parser.set_defaults(command='eval')
    train_parser.add_argument('--data', '-i', required=True, help='Data directory.')
    train_parser.add_argument('--filelimit', '-f', type=float, default=-1, help='limitation of sgf file')
    train_parser.add_argument('--processor', '-r', default="ValueProcessor", help='processor class')
    train_parser.add_argument('--evalmetric', '-m', default="acc", help='evaluate metric')
    train_parser.add_argument('--workers', '-w', type=int, default=-1, help='number of cpu worker')
    train_parser.add_argument('--batchsize', '-b', type=int, default=512, help='Batch size')
    train_parser.add_argument('--valuefile', '-v', help='model prefix')
    train_parser.add_argument('--valueepoch', '-a', type=int, default=1, help='model epoch')
    
   

    args = parser.parse_args()

    if args.command == 'eval':
      start_eval(args)
      

if __name__ == '__main__':
    main()

# command format
# python mxnet_value_evaluating.py eval --data data/eval --valuefile test_model_zoo/resnet_value --valueepoch 73
    