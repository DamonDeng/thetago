#!/bin/bash


python ./mxnet_dualres_training.py train \
--data data/ \
--network network.dual_resnet \
--prefix checkpoint/thetago_dual_res_17L_19res_0p001_win5_no_color_alllevel \
--epoche 100 \
--learningrate 0.001 \
--batchsize 256 \
--evalmetric mse \
--processor ZeroDualResProcessor \
--devices gpu \
--gpunumber 8 \
--levellimit 20k
