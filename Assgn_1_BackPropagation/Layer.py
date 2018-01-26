import numpy as np

class Layer:

    def __init__(self, size, activation):
        self.size = size
        self.activation = activation()
        if size == None:
            self.weights = None
            self.bias = None
        else:
            self.weights,self.bias = self.activation.init_weights(size)


    def feed_fwd(self,input):
        #print [input.shape,self.weights.shape]
        if self.weights is not None:
            net = np.matmul(input, self.weights) + self.bias
        else:
            net = input
        return self.activation(net)
