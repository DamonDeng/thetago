import mxnet as mx
import numpy as np
from data_loader.sgf_iter import SGFIter, SimulatorIter
from data_loader.multi_thread_sgf_iter import MultiThreadSGFIter
import logging
from data_loader.feature_processor import FeatureProcessor
from data_loader.original_processor import OriginalProcessor
from data_loader.value_processor import ValueProcessor
from robot.mxnet_robot import MXNetRobot

from sys import argv
import sys
import importlib
import os
import argparse
from multiprocessing import cpu_count

def start_training(args):

  print('self training starting 0.01')

  working_dir = args.dir
  model_prefix = args.prefix
  model_epoche = args.epoche
  
  devices = []
  if args.devices == 'gpu':
    for i in range(args.gpunumber):
      devices.append(mx.gpu(i))   
  else:
    devices = mx.cpu(0)

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

  

  learner_bot = MXNetRobot(model_prefix, model_epoche, processor)
  teacher_bot = MXNetRobot(model_prefix, model_epoche, processor)

  print('starting to play')
  for i in range(400):
    print('.'),
    position1 = learner_bot.select_move('b')
    if position1 is not None:
      teacher_bot.apply_move('b', position1)
    position2 = teacher_bot.select_move('w')
    if position2 is not None:
      learner_bot.apply_move('w', position2)

    if position1 is None and position2 is None:
      break
  
  print(learner_bot.get_board())

  (empty_score, black_core, white_score) = learner_bot.get_score()

  print ('Black:' + str(black_core) + ' White:' + str(white_score) + ' Empty:' + str(empty_score))

  learner_bot.analyst_result()

  

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
    train_parser.add_argument('--dir', required=True, help='Working directory.')
    train_parser.add_argument('--prefix', '-p', required=True, help='prefix of model.')
    train_parser.add_argument('--epoche', '-e', required=True, type=int, default=1, help='Number of epoche to load')
    train_parser.add_argument('--devices', '-d', default="cpu", help='prefix of checkpoint.')
    train_parser.add_argument('--batchsize', '-b', type=int, default=512, help='Batch size')
    train_parser.add_argument('--learningrate', '-l', type=float, default=0.1, help='Learning rate')
    train_parser.add_argument('--filelimit', '-f', type=float, default=-1, help='limitation of sgf file')
    train_parser.add_argument('--gpunumber', '-g', type=int, default=1, help='number of gpu')
    train_parser.add_argument('--processor', '-r', default="OriginalProcessor", help='processor class')
    train_parser.add_argument('--evalmetric', '-m', default="acc", help='evaluate metric')
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
# python self_training.py train --dir self_training/ws1 --prefix self_training/ws1/models/thetago_resnet3 --epoche 23

    