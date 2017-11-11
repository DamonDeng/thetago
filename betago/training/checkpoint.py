from __future__ import absolute_import
import os

import h5py
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.layers.core import Dense, Activation

from . import kerashack

import mxnet as mx

from data_loader.sgf_iter import SimulatorIter, SGFIter, DumyIter




__all__ = [
    'TrainingRun',
]


class TrainingRun(object):

    def __init__(self, filename, model, epochs_completed, chunks_completed, num_chunks):
        self.filename = filename
        self.model = model
        self.epochs_completed = epochs_completed
        self.chunks_completed = chunks_completed
        self.num_chunks = num_chunks

    def save(self):
        # Backup the original file in case something goes wrong while
        # saving the new checkpoint.
        backup = None
        if os.path.exists(self.filename):
            backup = self.filename + '.bak'
            os.rename(self.filename, backup)

        output = h5py.File(self.filename, 'w')
        metadata = output.create_group('metadata')
        metadata.attrs['epochs_completed'] = self.epochs_completed
        metadata.attrs['chunks_completed'] = self.chunks_completed
        metadata.attrs['num_chunks'] = self.num_chunks
        output.close()

        self.model.save_checkpoint(self.filename, 1)

        # If we got here, we no longer need the backup.
        if backup is not None:
            os.unlink(backup)

    def complete_chunk(self):
        self.chunks_completed += 1
        if self.chunks_completed == self.num_chunks:
            self.epochs_completed += 1
            self.chunks_completed = 0
        self.save()

    @classmethod
    def load(cls, filename):
        inp = h5py.File(filename, 'r')

        sym, arg_params, aux_params = mx.model.load_checkpoint(filename, 1)

        model = mx.mod.Module(symbol=sym, label_names=None, context=mx.cpu(0))
        # a urgly temp fix here, creating and empty SGFIter to get the data shape
        data_iter = DumyIter(batch_size=64, history_length=7)
        model.bind(for_training=True, data_shapes=data_iter.provide_data, label_shapes=data_iter.provide_label)
        model.set_params(arg_params, aux_params, allow_missing=True)


        # model = kerashack.load_model_from_hdf5_group(inp['model'])


        training_run = cls(filename,
                           model,
                           inp['metadata'].attrs['epochs_completed'],
                           inp['metadata'].attrs['chunks_completed'],
                           inp['metadata'].attrs['num_chunks'])
        inp.close()
        return training_run

    @classmethod
    def create(cls, filename, index, net, data_iter):
        
        devices = mx.cpu(0)
        # device = mx.gpu(0)

        model = mx.mod.Module(symbol=net,
                            context=devices)
        
        model.bind(data_shapes=(data_iter.provide_data), label_shapes=data_iter.provide_label)
        model.init_params()
        
        training_run = cls(filename, model, 0, 0, index.num_chunks)
        training_run.save()
        return training_run
