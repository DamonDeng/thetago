import mxnet as mx
import numpy as np
from data_loader.sgf_iter import SGFIter, SimulatorIter
import logging
from data_loader.feature_processor import FeatureProcessor
from data_loader.original_processor import OriginalProcessor
from sys import argv
import sys
import importlib
import os

def start_training():

  print('mxnet training starting 0.03')

  network_name = argv[1]
  data_dir = argv[2]
  checkpoint_prefix = argv[3]
  
  net = _load_network_by_name(network_name)

  logging.basicConfig(level=logging.INFO)
 
  processor = OriginalProcessor
  data_iter = SGFIter(sgf_directory=data_dir, batch_size=1024, file_limit = 2000, processor_class=processor)
  #data_iter = SimulatorIter( batch_size=1024)
  #data_iter = SimulatorIter(batch_size=1024, num_batches=1024)
 
  print (str(data_iter.provide_data))
  print (str(data_iter.provide_label))
  
  prefix = checkpoint_prefix

  devices = mx.cpu(0)
  # device = mx.gpu(0)

  mod = mx.mod.Module(symbol=net,
                      context=devices)

  mod.fit(data_iter, 
          num_epoch=1000, 
          eval_metric='ce',
          batch_end_callback=mx.callback.Speedometer(32, 20),
          epoch_end_callback=mx.callback.do_checkpoint(prefix))

def _load_module_from_filename(filename):
    if sys.version_info < (3, 3):
        import imp
        return imp.load_source('dynamicmodule', filename)
    elif sys.version_info < (3, 5):
        from importlib.machinery import SourceFileLoader
        return SourceFileLoader('dynamicmodule', filename).load_module()
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location('dynamicmodule', filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def _load_network_by_name(name):
    mod = None
    if os.path.exists(name):
        mod = _load_module_from_filename(name)
    else:
        try:
            mod = importlib.import_module('betago.' + name)
        except ImportError:
            mod = importlib.import_module(name)
    if not hasattr(mod, 'getSymbol'):
        raise ImportError('%s does not defined a layers function.' % (name,))
    return mod.getSymbol()




start_training()
