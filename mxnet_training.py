import mxnet as mx
import numpy as np
from data_loader.sgf_iter import SGFIter, SimulatorIter
from data_loader.multi_thread_sgf_iter import MultiThreadSGFIter
import logging
from data_loader.feature_processor import FeatureProcessor
from data_loader.original_processor import OriginalProcessor
from data_loader.value_processor import ValueProcessor
from data_loader.zero_processor import ZeroProcessor

from sys import argv
import sys
import importlib
import os
import argparse
from multiprocessing import cpu_count

def start_training(args):

  print('mxnet training starting 0.03')

  network_name = args.network
  data_dir = args.data
  checkpoint_prefix = args.prefix
  
  devices = []
  if args.devices == 'gpu':
    for i in range(args.gpunumber):
      devices.append(mx.gpu(i))
    
  else:
    devices = mx.cpu(0)
  

  net = _load_network_by_name(network_name)

  logging.basicConfig(level=logging.INFO)
  
  # Need automaticlly way to load the class
  if args.processor == 'FeatureProcessor':
    processor = FeatureProcessor
  elif args.processor == 'ValueProcessor':
    processor = ValueProcessor
  elif args.processor == 'OriginalProcessor':
    processor = OriginalProcessor
  elif args.processor == 'ZeroProcessor':
    processor = ZeroProcessor
  else:
    processor = OriginalProcessor

  workers = cpu_count()
  print('using '+ str(workers) + " workers to load the SGF data")
  data_iter = MultiThreadSGFIter(sgf_directory=data_dir, 
                                workers=workers, 
                                batch_size=args.batchsize, 
                                file_limit = args.filelimit, 
                                processor_class=processor, 
                                level_limit=args.levellimit)
  #data_iter = SimulatorIter( batch_size=1024)
  #data_iter = SimulatorIter(batch_size=1024, num_batches=1024)
 
  print (str(data_iter.provide_data))
  print (str(data_iter.provide_label))
  
  prefix = checkpoint_prefix

  # print(net.list_arguments())
  
  if args.oldmodel == 'keyword_none':
    mod = mx.mod.Module(symbol=net,
                        context=devices)
  else:
    
    mod = mx.mod.Module.load(args.oldmodel, args.oldepoche)


  try:
    print('started to traing....')
    mod.fit(data_iter, 
            num_epoch=args.epoche, 
            eval_metric=args.evalmetric,
            optimizer=args.optimizer,
            optimizer_params=(('learning_rate', args.learningrate),),
            batch_end_callback=mx.callback.Speedometer(32, 20),
            epoch_end_callback=mx.callback.do_checkpoint(prefix))
  finally:
    data_iter.stop_task()

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
    train_parser.add_argument('--optimizer', '-o', default="sgd", help='optimizer short name')
    train_parser.add_argument('--epoche', '-e', type=int, default=100, help='Number of epoche')
    train_parser.add_argument('--batchsize', '-b', type=int, default=512, help='Batch size')
    train_parser.add_argument('--learningrate', '-l', type=float, default=0.1, help='Learning rate')
    train_parser.add_argument('--filelimit', '-f', type=float, default=-1, help='limitation of sgf file')
    train_parser.add_argument('--gpunumber', '-g', type=int, default=1, help='number of gpu')
    train_parser.add_argument('--processor', '-r', default="OriginalProcessor", help='processor class')
    train_parser.add_argument('--evalmetric', '-m', default="acc", help='evaluate metric, acc, sme, ce')
    train_parser.add_argument('--levellimit', default="0d", help='player level limitation: xk,xd,xp')
    train_parser.add_argument('--player', default="all", help='player: all, winner, loser')
    train_parser.add_argument('--oldmodel', default="keyword_none", help='the old model you want to load')
    train_parser.add_argument('--oldepoche', default=0, type=int, help='the epoche of old model')


    args = parser.parse_args()

    if args.command == 'train':
      start_training(args)
      

if __name__ == '__main__':
    main()

# command format
# python mxnet_training.py train --data data/standard --network network.original_cnn --prefix checkpoint/testing12 --learningrate 2
# python mxnet_training.py train --data data --network network.resnet --prefix checkpoint/zero_resnet --learningrate 0.1 --processor ZeroProcessor --filelimit 10 --levellimit 0d --batchsize 16 --evalmetric ce
# python mxnet_training.py train --data data --network network.resnet --prefix checkpoint/zero_resnet --learningrate 0.1 --processor ZeroProcessor --filelimit 50 --levellimit 0d --device gpu --gpunumber 8 --evalmetric ce --batchsize 2056
    