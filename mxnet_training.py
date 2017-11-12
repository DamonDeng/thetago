import mxnet as mx
import numpy as np
from data_loader.sgf_iter import SGFIter, SimulatorIter
import logging
from data_loader.feature_processor import FeatureProcessor


def simple_CNN():

  print('mxnet training with CNN  0.02')
  
  num_classes = 361

  data = mx.symbol.Variable('data')
  # first conv layer
  conv1 = mx.sym.Convolution(data=data, kernel=(7,7), num_filter=48)
  tanh1 = mx.sym.Activation(data=conv1, act_type="relu")
  # pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
  # second conv layer
  conv2 = mx.sym.Convolution(data=tanh1, kernel=(5,5), num_filter=32)
  tanh2 = mx.sym.Activation(data=conv2, act_type="relu")
  # pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
  # first fullc layer
  conv3 = mx.sym.Convolution(data=tanh2, kernel=(5,5), num_filter=32)
  tanh3 = mx.sym.Activation(data=conv3, act_type="relu")
  
  flatten = mx.sym.Flatten(data=tanh3)
  fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512)
  tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
  # second fullc
  fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=361)
  # softmax loss
  cnn_net = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

  logging.basicConfig(level=logging.INFO)
 
  processor = FeatureProcessor
  data_iter = SGFIter(sgf_directory='data/standard', batch_size=1024, file_limit = 2000, processor_class=processor)
  #data_iter = SimulatorIter( batch_size=1024)
  #data_iter = SimulatorIter(batch_size=1024, num_batches=1024)
 
  print (str(data_iter.provide_data))
  print (str(data_iter.provide_label))
  
  prefix = 'checkpoint/thetago_testing3'

  devices = mx.cpu(0)
  # device = mx.gpu(0)

  mod = mx.mod.Module(symbol=cnn_net,
                      context=devices)

  mod.fit(data_iter, 
          num_epoch=1000, 
          eval_metric='ce',
          batch_end_callback=mx.callback.Speedometer(32, 20),
          epoch_end_callback=mx.callback.do_checkpoint(prefix))

simple_CNN()
