import mxnet as mx
import numpy as numpy

class DualMetricCE(mx.metric.EvalMetric): 
    def __init__(self, num=None):
        super(DualMetricCE, self).__init__('Dual_CE', num)

    def update(self, labels, preds):
        labels, preds = mx.metric.check_label_shapes(labels, preds, True)

        

        for label, pred in zip(labels[0], preds[0]):


            
            label = label.asnumpy()
            pred = pred.asnumpy()

            

            label = label.ravel().astype(dtype='int32')

            # print ('label shape:')
            # print (label.shape[0])
            # print ('pred shape:')
            # print (pred.shape[0])

            # assert label.shape[0] == pred.shape[0]

            # print ('label:')
            # print (label)
            # print ('-------------------------')
            # print ('preds:')
            # print (pred)
            # print ('----------------------------------------------')

            prob = pred[label[0]]

            self.sum_metric += (-numpy.log(prob + 0.0000000001))
            self.num_inst += 1


class DualMetricMSE(mx.metric.EvalMetric): 
    def __init__(self, num=None):
        super(DualMetricMSE, self).__init__('Dual_MSE', num)

    def update(self, labels, preds):
        labels, preds = mx.metric.check_label_shapes(labels, preds, True)

        for label, pred in zip(labels[1], preds[1]):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)
            if len(pred.shape) == 1:
                pred = pred.reshape(pred.shape[0], 1)

            self.sum_metric += ((label - pred)**2.0).mean()
            self.num_inst += 1 # numpy.prod(label.shape)