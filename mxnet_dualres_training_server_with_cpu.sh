#!/bin/bash


python ./mxnet_dualres_training.py train \
--data data/ \
--network network.dual_resnet \
--prefix checkpoint/thetago_dual_res_17L_19res \
--epoche 100 \
--learningrate 0.1 \
--batchsize 16 \
--evalmetric mse \
--processor ZeroDualResProcessor 



