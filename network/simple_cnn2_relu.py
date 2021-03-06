import mxnet as mx

def getSymbol():
  num_classes = 361

  data = mx.symbol.Variable('data')
  # first conv layer
  conv1 = mx.sym.Convolution(data=data, kernel=(3,3), num_filter=48)
  tanh1 = mx.sym.Activation(data=conv1, act_type="relu")
  # pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
  # second conv layer
  conv2 = mx.sym.Convolution(data=tanh1, kernel=(3,3), num_filter=32)
  tanh2 = mx.sym.Activation(data=conv2, act_type="relu")
  # pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
  # first fullc layer
  conv3 = mx.sym.Convolution(data=tanh2, kernel=(3,3), num_filter=32)
  tanh3 = mx.sym.Activation(data=conv3, act_type="relu")
  
  flatten = mx.sym.Flatten(data=tanh3)
  fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512)
  tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
  # second fullc
  fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=361)
  # softmax loss
  cnn_net = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

  return cnn_net