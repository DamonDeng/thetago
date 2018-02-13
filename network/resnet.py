import mxnet as mx

def getSymbol():
  num_classes = 362
  bn_mom=0.9

  data = mx.symbol.Variable('data')
  conv1 = mx.sym.Convolution(data=data, kernel=(3,3), num_filter=256, pad = (1,1))
  bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom)
  acti1 = mx.sym.Activation(data=bn1, act_type="relu")
  
  res_net = acti1
  for i in range(19):
    res_conv1 = mx.sym.Convolution(data=res_net, kernel=(3,3), num_filter=256, pad = (1,1))
    res_bn1 = mx.sym.BatchNorm(data=res_conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    res_acti1 = mx.sym.Activation(data=res_bn1, act_type="relu")
    
    res_conv2 = mx.sym.Convolution(data=res_acti1, kernel=(3,3), num_filter=256, pad = (1,1))
    res_bn2 = mx.sym.BatchNorm(data=res_conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    res_acti2 = mx.sym.Activation(data=res_bn2, act_type="relu")

    temp_result = res_net + res_acti2
    res_net = temp_result
  
  flatten = mx.sym.Flatten(data=res_net)
  fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512)
  tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
  fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=num_classes)
  final_net = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

  return final_net