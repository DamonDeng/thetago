import mxnet as mx
import numpy as np
from data_loader.sgf_iter import SGFIter, SimulatorIter
import logging
from data_loader.feature_processor import FeatureProcessor
from data_loader.original_processor import OriginalProcessor
from sys import argv
import sys
import importlib
import os
import argparse

def start_training(args):

  print('mxnet training starting 0.03')

  network_name = args.network
  data_dir = args.data
  checkpoint_prefix = args.prefix
  
  if args.devices == 'gpu':
    devices = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3), mx.gpu(4), mx.gpu(5), mx.gpu(6), mx.gpu(7)] 
  else:
    devices = mx.cpu(0)
  

  net = _load_network_by_name(network_name)

  logging.basicConfig(level=logging.INFO)
 
  processor = OriginalProcessor
  data_iter = SGFIter(sgf_directory=data_dir, batch_size=1024, file_limit = 2000, processor_class=processor)
  #data_iter = SimulatorIter( batch_size=1024)
  #data_iter = SimulatorIter(batch_size=1024, num_batches=1024)
 
  print (str(data_iter.provide_data))
  print (str(data_iter.provide_label))
  
  prefix = checkpoint_prefix

  
  mod = mx.mod.Module(symbol=net,
                      context=devices)

  mod.fit(data_iter, 
          num_epoch=args.epoche, 
          eval_metric='ce',
          optimizer_params=(('learning_rate', args.learningrate),),
          batch_end_callback=mx.callback.Speedometer(32, 20),
          epoch_end_callback=mx.callback.do_checkpoint(prefix))

def _load_module_from_filename(filename):
    if sys.version_info < (3, 3):
        import imp
        return imp.load_source('dynamicmodule', filename)
    elif sys.version_info < (3, 5):
        from importlib.machinery import SourceFileLoader
        return SourceFileLoader('dynamicmodule', filename).load_module()
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location('dynamicmodule', filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def _load_network_by_name(name):
    mod = None
    if os.path.exists(name):
        mod = _load_module_from_filename(name)
    else:
        try:
            mod = importlib.import_module('betago.' + name)
        except ImportError:
            mod = importlib.import_module(name)
    if not hasattr(mod, 'getSymbol'):
        raise ImportError('%s does not defined a layers function.' % (name,))
    return mod.getSymbol()




def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train', help='Do some training.')
    train_parser.set_defaults(command='train')
    train_parser.add_argument('--data', '-i', required=True, help='Data directory.')
    train_parser.add_argument('--network', '-n', required=True, help='Network to use.')
    train_parser.add_argument('--prefix', '-p', required=True, help='prefix of checkpoint.')
    train_parser.add_argument('--devices', '-d', default="cpu", help='prefix of checkpoint.')
    train_parser.add_argument('--epoche', '-e', type=int, default=100, help='Number of epoche')
    train_parser.add_argument('--learningrate', '-l', type=float, default=0.1, help='Learning rate')


    args = parser.parse_args()

    if args.command == 'train':
      start_training(args)
      

if __name__ == '__main__':
    main()

# command format
# python mxnet_training.py train --data data/standard --network network.original_cnn --prefix checkpoint/testing12 --learningrate 2
    