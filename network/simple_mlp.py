import mxnet as mx

def getSymbol():
  num_classes = 361
  net = mx.sym.Variable('data')
  # net = mx.sym.Flatten(data=net, name='flatten')
  net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=361)
  net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
  net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=num_classes)
  net = mx.sym.SoftmaxOutput(data=net, name='softmax')

  return net