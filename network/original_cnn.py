import mxnet as mx

def getSymbol():
  num_classes = 361

  data = mx.symbol.Variable('data')
  # first conv layer
  conv1 = mx.sym.Convolution(data=data, kernel=(7,7), num_filter=64, pad = (3,3))
  acti1 = mx.sym.Activation(data=conv1, act_type="relu")
  # pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
  # second conv layer
  conv2 = mx.sym.Convolution(data=acti1, kernel=(5,5), num_filter=64, pad = (2,2))
  acti2 = mx.sym.Activation(data=conv2, act_type="relu")
  # pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
  # first fullc layer
  conv3 = mx.sym.Convolution(data=acti2, kernel=(5,5), num_filter=48, pad = (2,2))
  acti3 = mx.sym.Activation(data=conv3, act_type="relu")

  conv4 = mx.sym.Convolution(data=acti3, kernel=(5,5), num_filter=48, pad = (2,2))
  acti4 = mx.sym.Activation(data=conv4, act_type="relu")

  conv5 = mx.sym.Convolution(data=acti4, kernel=(5,5), num_filter=48, pad = (2,2))
  acti5 = mx.sym.Activation(data=conv5, act_type="relu")

  conv6 = mx.sym.Convolution(data=acti5, kernel=(5,5), num_filter=32, pad = (2,2))
  acti6 = mx.sym.Activation(data=conv6, act_type="relu")

  conv7 = mx.sym.Convolution(data=acti6, kernel=(5,5), num_filter=32, pad = (2,2))
  acti7 = mx.sym.Activation(data=conv7, act_type="relu")

  
  flatten = mx.sym.Flatten(data=acti7)
  fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=1024)
  acti_fc1 = mx.sym.Activation(data=fc1, act_type="relu")
  # second fullc
  fc2 = mx.sym.FullyConnected(data=acti_fc1, num_hidden=361)
  # softmax loss
  cnn_net = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

  return cnn_net