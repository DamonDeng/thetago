import mxnet as mx
import numpy as np
from data_loader.sgf_iter import SimulatorIter, SGFIter
import logging


def simple_CNN():

  print('mxnet training with CNN  0.01')
  
  num_classes = 361

  data = mx.symbol.Variable('data')
  # first conv layer
  conv1 = mx.sym.Convolution(data=data, kernel=(4,4), pad=(1,1), num_filter=20)
  acti1 = mx.sym.Activation(data=conv1, act_type="relu")
  pool1 = mx.sym.Pooling(data=acti1, pool_type="max", kernel=(2,2), stride=(1,1))
  # second conv layer
  conv2 = mx.sym.Convolution(data=pool1, kernel=(4,4), pad=(1,1), num_filter=50)
  acti2 = mx.sym.Activation(data=conv2, act_type="relu")
  pool2 = mx.sym.Pooling(data=acti2, pool_type="max", kernel=(2,2), stride=(1,1))
  # first fullc layer
  conv3 = mx.sym.Convolution(data=pool2, kernel=(4,4), num_filter=70)
  acti3 = mx.sym.Activation(data=conv3, act_type="relu")
  pool3 = mx.sym.Pooling(data=acti3, pool_type="max", kernel=(2,2), stride=(2,2))
  
  flatten = mx.sym.Flatten(data=pool3)
  fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
  acti3 = mx.sym.Activation(data=fc1, act_type="tanh")
  # second fullc
  fc2 = mx.sym.FullyConnected(data=acti3, num_hidden=361)
  # softmax loss
  cnn_net = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

  logging.basicConfig(level=logging.INFO)
 
  data_iter = SGFIter(sgf_directory='data/train', batch_size=1024, file_limit = -1)
  #data_iter = SimulatorIter(batch_size=1024, num_batches=30)
  print (str(data_iter.provide_data))
  print (str(data_iter.provide_label))
  
  prefix = 'checkpoint/thetago_CNN8'

  #devices = mx.cpu(0)
  # devices = mx.gpu(0)
  devices = [mx.gpu(0),mx.gpu(1),mx.gpu(2),mx.gpu(3),mx.gpu(4),mx.gpu(5),mx.gpu(6),mx.gpu(7)]

  mod = mx.mod.Module(symbol=cnn_net,
                      context=devices)

  mod.fit(data_iter, 
          num_epoch=100, 
          batch_end_callback=mx.callback.Speedometer(32, 20),
          epoch_end_callback=mx.callback.do_checkpoint(prefix))

simple_CNN()
