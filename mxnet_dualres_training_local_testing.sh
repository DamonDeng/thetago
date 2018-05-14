#!/bin/bash


python ./mxnet_dualres_training_testing.py train \
--data data/small_tar/ \
--eval data/small_eval/ \
--network network.dual_resnet \
--prefix checkpoint/thetago_dual_res \
--epoche 100 \
--learningrate 0.1 \
--batchsize 4 \
--evalmetric mse \
--processor ZeroDualResProcessor \
> ./log/training_log.txt 2>&1 &


