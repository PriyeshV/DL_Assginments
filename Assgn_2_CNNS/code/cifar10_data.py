from __future__ import generators
import collections
import numpy as np
from utils import data_iterator
import cPickle as pickle
import numpy as np

class DataSet(object):
    def __init__(self, x, y):
        self._x = x/255.
        self._y = y
        self._n_samples = x.shape[0]
        self._n_features = x.shape[1]
        self._n_classes = y.shape[1]

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set.
           Takes Y in one-hot encodings only"""
        for x, y in data_iterator(self._x, self._y, batch_size, shuffle):
            yield x, y

def read_data_sets(data_dir, val_size=5000):

    data = np.zeros((50000,3072))
    labels = np.zeros((50000,10))

    for i in range(1,6):
        f_name = data_dir+'data_batch_'+str(i)
        fo = open(f_name, 'rb')
        dict = pickle.load(fo)
        data[(i-1)*10000:i*10000,:] = dict['data']

        tlabels = np.zeros((10000,10))
        tlabels[np.arange(10000),dict['labels']] = 1
        labels[(i-1)*10000:i*10000,:] = tlabels
        fo.close()

    np.random.seed(12)
    index = np.random.permutation(np.arange(50000))

    tr_data = data[index[:-val_size],:]
    tr_labels = labels[index[:-val_size],:]
    val_data = data[index[val_size:],:]
    val_labels = labels[index[val_size:],:]

    f_name = data_dir + 'test_batch'
    fo = open(f_name, 'rb')
    dict = pickle.load(fo)
    tlabels = np.zeros((10000, 10))
    tlabels[np.arange(10000), dict['labels']] = 1

    te_data = dict['data']
    te_labels = tlabels

    train = DataSet(tr_data, tr_labels)
    validation = DataSet(val_data, val_labels)
    test = DataSet(te_data, te_labels)

    datasets_template = collections.namedtuple('Datasets_template', ['train','validation','test'])
    Datasets = datasets_template(train=train, validation=validation, test=test)

    return Datasets

def load_data():
  return read_data_sets(data_dir='cifar-10-batches-py/')