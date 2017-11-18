#!/bin/bash

python ./mxnet_training.py train --data data/train --network network.resnet --prefix checkpoint/thetago_resnet3 --epoche 100 --learningrate 0.1 --batchsize 1024 --devices gpu --gpunumber 8 --evalmetric ce