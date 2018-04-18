import mxnet as mx

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
  for i in range(19):
    res_conv1 = mx.sym.Convolution(data=res_net, kernel=(3,3), num_filter=256, pad = (1,1))
    res_bn1 = mx.sym.BatchNorm(data=res_conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    res_acti1 = mx.sym.Activation(data=res_bn1, act_type="relu")
    
    res_conv2 = mx.sym.Convolution(data=res_acti1, kernel=(3,3), num_filter=256, pad = (1,1))
    res_bn2 = mx.sym.BatchNorm(data=res_conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    res_acti2 = mx.sym.Activation(data=res_bn2, act_type="relu")

    temp_result = res_net + res_acti2
    res_net = temp_result

  policy_conv = mx.sym.Convolution(data=res_net, kernel=(1,1), num_filter=2)
  policy_bn = mx.sym.BatchNorm(data=policy_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom)
  policy_acti = mx.sym.Activation(data=policy_bn, act_type='relu')
  policy_fc = mx.sym.FullyConnected(data=policy_acti, num_hidden=num_classes)
  policy_output = mx.sym.SoftmaxOutput(data=policy_fc, name='move', label=move_label)

  value_conv = mx.sym.Convolution(data=res_net, kernel=(1,1), num_filter=1)
  value_bn = mx.sym.BatchNorm(data=value_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom)
  value_acti1 = mx.sym.Activation(data=value_bn, act_type='relu')
  value_fc1 = mx.sym.FullyConnected(data=value_acti1, num_hidden=256)
  value_acti2 = mx.sym.Activation(data=value_fc1, act_type='relu')
  value_fc2 = mx.sym.FullyConnected(data=value_acti2, num_hidden=1)

  value_acti2 = mx.sym.Activation(data=value_fc2, act_type="tanh")

  value_output = mx.sym.LinearRegressionOutput(data=value_acti2, label=value_label)

  final_net = mx.symbol.Group([policy_output, value_output])


  return final_net