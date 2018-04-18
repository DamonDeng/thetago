import mxnet as mx
import numpy as np
from data_loader.sgf_iter import SGFIter, SimulatorIter
from data_loader.multi_thread_dual_sgf_iter import MultiThreadDualSGFIter
import logging
from data_loader.feature_processor import FeatureProcessor
from data_loader.original_processor import OriginalProcessor
from data_loader.value_processor import ValueProcessor
from data_loader.zero_processor import ZeroProcessor
from data_loader.zero_dualres_processor import ZeroDualResProcessor

from sys import argv
import sys
import importlib
import os
import argparse
from multiprocessing import cpu_count

def getSymbol():
  num_classes = 362
  bn_mom=0.9

  data = mx.symbol.Variable('data')
  value_label = mx.symbol.Variable('value_label')
  move_label = mx.symbol.Variable('move_label')
  
  conv1 = mx.sym.Convolution(data=data, kernel=(3,3), num_filter=256, pad = (1,1))
  bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom)
  acti1 = mx.sym.Activation(data=bn1, act_type="relu")
  
  res_net = acti1
  # for i in range(19):
  #   res_conv1 = mx.sym.Convolution(data=res_net, kernel=(3,3), num_filter=256, pad = (1,1))
  #   res_bn1 = mx.sym.BatchNorm(data=res_conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom)
  #   res_acti1 = mx.sym.Activation(data=res_bn1, act_type="relu")
    
  #   res_conv2 = mx.sym.Convolution(data=res_acti1, kernel=(3,3), num_filter=256, pad = (1,1))
  #   res_bn2 = mx.sym.BatchNorm(data=res_conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom)
  #   res_acti2 = mx.sym.Activation(data=res_bn2, act_type="relu")

  #   temp_result = res_net + res_acti2
  #   res_net = temp_result

  policy_conv = mx.sym.Convolution(data=res_net, kernel=(1,1), num_filter=2)
  policy_bn = mx.sym.BatchNorm(data=policy_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom)
  policy_acti = mx.sym.Activation(data=policy_bn, act_type='relu')
  policy_fc = mx.sym.FullyConnected(data=policy_acti, num_hidden=362)
  policy_output = mx.sym.SoftmaxOutput(data=policy_fc, name='softmax', label=move_label)

  value_conv = mx.sym.Convolution(data=res_net, kernel=(1,1), num_filter=1)
  value_bn = mx.sym.BatchNorm(data=value_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom)
  value_acti1 = mx.sym.Activation(data=value_bn, act_type='relu')
  value_fc1 = mx.sym.FullyConnected(data=value_acti1, num_hidden=256)
  value_acti2 = mx.sym.Activation(data=value_fc1, act_type='relu')
  value_fc2 = mx.sym.FullyConnected(data=value_acti2, num_hidden=1)

  value_acti2 = mx.sym.Activation(data=value_fc2, act_type="tanh")

  value_output = mx.sym.LinearRegressionOutput(data=value_acti2, label=value_label)

  final_result = mx.sym.Group([policy_output, value_output])


  return final_result


class SimpleIter(mx.io.DataIter):
    def __init__(self, data_names, data_shapes, data_gen,
                 label_names, label_shapes, label_gen, num_batches=10):
        self._provide_data = list(zip(data_names, data_shapes))
        self._provide_label = list(zip(label_names, label_shapes))
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
#         temp = [ str(d[1]) for d,g in zip(self._provide_label, self.label_gen)]
#         print (temp[1])
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            # print('label inside the generator:')
            # print(label)
            # print('item 0:')
            # print(label[0])
            # print('item 1:')
            # print(label[1])
            
            return mx.io.DataBatch(data, label)
        else:
            raise StopIteration






def start_training(args):

  print('mxnet dual res net training starting 0.01')

  network_name = args.network
  data_dir = args.data
  checkpoint_prefix = args.prefix
  
  devices = []
  if args.devices == 'gpu':
    for i in range(args.gpunumber):
      devices.append(mx.gpu(i))
    
  else:
    devices = mx.cpu(0)
  

  # net = _load_network_by_name(network_name)

  net = getSymbol()

  logging.basicConfig(level=logging.INFO)
  
  # Need automaticlly way to load the class
  print('using processor: ' + args.processor)
  if args.processor == 'FeatureProcessor':
    processor = FeatureProcessor
  elif args.processor == 'ValueProcessor':
    processor = ValueProcessor
  elif args.processor == 'OriginalProcessor':
    processor = OriginalProcessor
  elif args.processor == 'ZeroProcessor':
    processor = ZeroProcessor
  elif args.processor == 'ZeroDualResProcessor':
    processor = ZeroDualResProcessor
  else:
    processor = OriginalProcessor

  print('using processor :::::::: ' + str(processor))

  workers = cpu_count()
  print('using '+ str(workers) + " workers to load the SGF data")
  data_iter = MultiThreadDualSGFIter(sgf_directory=data_dir, 
                                workers=workers, 
                                batch_size=args.batchsize, 
                                file_limit = args.filelimit, 
                                processor_class=processor, 
                                level_limit=args.levellimit)
  #data_iter = SimulatorIter( batch_size=1024)
  #data_iter = SimulatorIter(batch_size=1024, num_batches=1024)


  # n = 32
  # num_classes = 362
  # data_iter = SimpleIter(['data'], [(n, 17, 19, 19)],
  #                 [lambda s: np.random.uniform(-1, 1, s)],
  #                 ['move_label', 'value_label'], [(n, num_classes), (n,)],
  #                 [lambda s: np.random.randint(0, 3, s), lambda s: np.random.randint(0, 3, s)])
 
  print (str(data_iter.provide_data))
  print (str(data_iter.provide_label))
  
  prefix = checkpoint_prefix

  # print(net.list_arguments())
  
  if args.oldmodel == 'keyword_none':
    mod = mx.mod.Module(symbol=net,
                        context=devices,
                        label_names=['move_label', 'value_label'])
  else:
    
    mod = mx.mod.Module.load(args.oldmodel, args.oldepoche)


  try:
    print('started to train....')


    # print('debug display for iter:')
    # temp_iter = data_iter.next()

    # print ('iter:' + str(temp_iter))

    # print ('end of debug displaying')

    # print ('the label is:')
    # print (temp_iter.label)

    mod.fit(data_iter, 
            num_epoch=args.epoche, 
            eval_metric=args.evalmetric,
            optimizer=args.optimizer,
            optimizer_params=(('learning_rate', args.learningrate),),
            batch_end_callback=mx.callback.Speedometer(32, 20),
            epoch_end_callback=mx.callback.do_checkpoint(prefix))

    # mod.fit(data_iter, 
    #         num_epoch=5, 
    #         eval_metric='mse',
    #         optimizer='sgd',
    #         optimizer_params=(('learning_rate', 0.1),),
    #         batch_end_callback=mx.callback.Speedometer(32, 20),
    #         epoch_end_callback=mx.callback.do_checkpoint('dual_res_notebook_testing_new'))


  finally:
    # data_iter.stop_task()
    print ('finally code segment')

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
    