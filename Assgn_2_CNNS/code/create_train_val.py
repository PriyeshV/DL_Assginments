from __future__ import generators
import os
import cPickle as pickle
import numpy as np

data_dir='cifar-10-batches-py/'
new_dir='dataset/'

if not os.path.exists(new_dir):
    os.makedirs(new_dir)

for i in range(1, 6):
    f_name = data_dir + 'data_batch_' + str(i)
    fo = open(f_name, 'rb')
    dict = pickle.load(fo)
    fo.close()

    data = dict['data'].astype(dtype=np.uint8)
    labels = np.zeros((10000, 10),dtype=np.uint8)
    labels[np.arange(10000), dict['labels']] = 1

    if i == 5:
        train = np.concatenate((labels[:5000,:], data[:5000,:]), axis=1)
        train.tofile(new_dir+'train_batch_'+str(i)+'.bin')
        validation = np.concatenate((labels[5000:,:], data[5000:,:]), axis=1)
        validation.tofile(new_dir + 'val_batch_' + str(1)+'.bin')
        break
    data = np.concatenate((labels, data), axis=1)
    data.tofile(new_dir+'train_batch_'+str(i)+'.bin')

f_name = data_dir + 'test_batch'
fo = open(f_name, 'rb')
dict = pickle.load(fo)
fo.close()

data = dict['data'].astype(dtype=np.uint8)
labels = np.zeros((10000, 10),dtype=np.uint8)
labels[np.arange(10000), dict['labels']] = 1
data = np.concatenate((labels, data), axis=1)
data.tofile('{0}test_batch_{1}.bin'.format(new_dir, str(1)))