#!/bin/bash

python ./mxnet_training.py train --data data/train --network network.original_cnn --prefix checkpoint/thetago_r1 --epoche 1000 --learningrate 0.5 --batchsize 1024