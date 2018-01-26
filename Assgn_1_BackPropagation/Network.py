import numpy as np
import cPickle, gzip
from Layer import Layer
from Activation import identity,sigmoid,relu,tanh,leaky_relu
from cost import sq,ce

class Network:

    def __init__(self, args,n_features,n_labels):

        self.args = args
        self.n_features = n_features
        self.n_labels = n_labels

        self.shapes = [int(shape) for shape in self.args.sizes.split(",")]
        self.n_hidden = len(self.shapes)
        self.n_layers = self.n_hidden + 3
        # 0: input layer | 1-n_hidden: hidden layers | output layer | cost_layer
        self.layers = {}

        self.create_layers()
        #self.layers[self.n_layers] = Layer(self.laters[0].shape,'ce_loss')

        # Weight matrices are in between the layers.
        #assert len(self.shapes) == len(self.layers) - 1

    def create_layers(self):

        self.shapes.insert(0,self.n_features)
        self.shapes.append(self.n_labels)
        self.shapes.append(self.n_labels)
        self.weight_dims = zip(self.shapes[:-1], self.shapes[1:])

        #Layers start from 1 - input layer is 0
        #Hidden Layers
        for i in xrange(1,self.n_layers-2):
            self.layers[i] = Layer(self.weight_dims[i-1], globals()[self.args.activation])
        #Output Layer
        self.layers[i+1] = Layer(self.weight_dims[i],sigmoid)
        #Loss Layer
        self.layers[i+2] = globals()[self.args.loss]()

    def save_network(self):
        pass

    def load_network(self):
        pass

    def one_step_history(self):
        pass

    def one_step_lookahead(self):
        pass