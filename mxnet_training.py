import mxnet as mx
import numpy as np
from data_loader.sgf_iter import SimpleIter, SimulatorIter, SGFIter
import logging


def simple_mlp():

  print('mxnet training with MLP  0.01')
  
  num_classes = 361
  net = mx.sym.Variable('data')
  net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=64)
  net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
  net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=num_classes)
  net = mx.sym.SoftmaxOutput(data=net, name='softmax')
  print(net.list_arguments())
  print(net.list_outputs())




  logging.basicConfig(level=logging.INFO)

 
  data_iter = SGFIter(sgf_directory='data/train', batch_size=32, file_limit = 200)
 
 
  print (str(data_iter.provide_data))
  print (str(data_iter.provide_label))

  c = data_iter.next()

  print (str(c))
  prefix = 'checkpoint/mxnet_thetago'
  mod = mx.mod.Module(symbol=net)
  mod.fit(data_iter, 
          num_epoch=100, 
          batch_end_callback=mx.callback.Speedometer(32, 20),
          epoch_end_callback=mx.callback.do_checkpoint(prefix))

def simple_CNN():

  print('mxnet training with CNN  0.01')
  
  num_classes = 361

  data = mx.symbol.Variable('data')
  # first conv layer
  conv1 = mx.sym.Convolution(data=data, kernel=(3,3), num_filter=20)
  tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
  # pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
  # second conv layer
  conv2 = mx.sym.Convolution(data=tanh1, kernel=(3,3), num_filter=50)
  tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
  # pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
  # first fullc layer
  conv3 = mx.sym.Convolution(data=tanh2, kernel=(3,3), num_filter=70)
  tanh3 = mx.sym.Activation(data=conv3, act_type="tanh")
  
  flatten = mx.sym.Flatten(data=tanh3)
  fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
  tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
  # second fullc
  fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=361)
  # softmax loss
  cnn_net = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

  logging.basicConfig(level=logging.INFO)
 
  data_iter = SGFIter(sgf_directory='data/train', batch_size=32, file_limit = 20)
 
  print (str(data_iter.provide_data))
  print (str(data_iter.provide_label))
  
  prefix = 'checkpoint/mxnet_thetago_CNN'

  devices = mx.cpu(0)
  # device = mx.gpu(0)

  mod = mx.mod.Module(symbol=cnn_net,
                      context=devices)

  mod.fit(data_iter, 
          num_epoch=100, 
          batch_end_callback=mx.callback.Speedometer(32, 20),
          epoch_end_callback=mx.callback.do_checkpoint(prefix))

simple_CNN()