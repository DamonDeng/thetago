#!/bin/bash


python ./mxnet_dualres_training.py train \
--data data/small_tar/ \
--network network.dual_resnet \
--prefix checkpoint/thetago_dual_res \
--epoche 100 \
--learningrate 0.1 \
--batchsize 4 \
--evalmetric mse \
--processor ZeroDualResProcessor


