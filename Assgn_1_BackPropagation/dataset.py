import numpy as np

class DataSet(object):

  def __init__(self,x,y):
    """Construct a DataSet.
    """
    self.x = x
    self.y = y
    self.n_samples = x.shape[0]
    self.n_features = x.shape[1]
    self.n_labels = 10

  def next_batch(self,batch_size=30, shuffle=False):
      # Optionally shuffle the data before training
      # pulled from cs224d assignments and modified
      if shuffle:
          indices = np.random.permutation(self.n_samples)
          data_X = self.x[indices]
          data_y = self.y[indices]
      else:
          data_X = self.x
          data_y = self.y
      ###
      total_processed_examples = 0
      total_steps = int(len(data_X) // batch_size)
      for step in xrange(total_steps):
          # Create the batch by selecting up to batch_size elements
          batch_start = step * batch_size
          x = data_X[batch_start:batch_start + batch_size]
          y = data_y[batch_start:batch_start + batch_size]
          yield x, y
          total_processed_examples += len(x)

      # Sanity check to make sure we iterated over all the dataset as intended
      assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X),
                                                                                            total_processed_examples)