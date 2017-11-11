import mxnet as mx
import numpy as np
from data_loader.sgf_iter import SimulatorIter, SGFIter
import logging

from sys import argv


def simple_CNN_continue(old_prefix, old_epo_number, new_prefix):

  print('mxnet continue training with CNN  0.01')
  
  sym, arg_params, aux_params = mx.model.load_checkpoint(old_prefix, old_epo_number)
  devices = mx.cpu(0)
  #devices = [mx.gpu(0),mx.gpu(1),mx.gpu(2),mx.gpu(3),mx.gpu(4),mx.gpu(5),mx.gpu(6),mx.gpu(7)]
  mod = mx.mod.Module(symbol=sym, label_names=None, context=devices)
  mod.bind(for_training=True, data_shapes=mod._data_shapes, label_shapes=mod._label_shapes)
  mod.set_params(arg_params, aux_params, allow_missing=True)


  logging.basicConfig(level=logging.INFO)
 
  data_iter = SGFIter(sgf_directory='data/standard', batch_size=16, file_limit = 2000)
  # data_iter = SimulatorIter( batch_size=1024)
  #data_iter = SimulatorIter(batch_size=64, num_batches=1024)
 
  print (str(data_iter.provide_data))
  print (str(data_iter.provide_label))
  
  prefix = new_prefix

  devices = mx.cpu(0)
  # device = mx.gpu(0)

  # mod = mx.mod.Module(symbol=cnn_net,
  #                     context=devices)

  mod.fit(data_iter, 
          num_epoch=100, 
          eval_metric='ce',
          batch_end_callback=mx.callback.Speedometer(32, 20),
          epoch_end_callback=mx.callback.do_checkpoint(prefix))

simple_CNN_continue(argv[1], argv[2], argv[3])
