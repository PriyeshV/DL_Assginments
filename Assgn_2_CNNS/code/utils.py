from collections import defaultdict
import numpy as np


def data_iterator(orig_X, orig_y, batch_size=32, shuffle=False):
    if shuffle:
        indices = np.random.permutation(len(orig_X))
        data_X = orig_X[indices]
        data_y = orig_y[indices] if np.any(orig_y) else None
    else:
        data_X = orig_X
        data_y = orig_y
    ###

    total_processed_examples = 0
    total_steps = int(len(data_X) // batch_size)
    for step in xrange(total_steps):
        # Create the batch by selecting up to batch_size elements
        batch_start = step * batch_size
        x = data_X[batch_start:batch_start + batch_size]
        y = data_y[batch_start:batch_start + batch_size]

        yield x, y
    #total_processed_examples += len(x)
    # Sanity check to make sure we iterated over all the dataset as intended
#    assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X),total_processed_examples)


def labels_to_onehot(data_y, num_instances, num_labels):
    # assuming the labes start with 0
    y = np.zeros((num_instances, num_labels), dtype=np.int32)
    y[np.arange(num_instances, dtype=np.int32), data_y] = 1
    return y
