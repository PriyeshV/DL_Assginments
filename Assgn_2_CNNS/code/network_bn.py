import tensorflow as tf
import numpy as np
import math

#class Network(Architecture):
class Network(object):

    def __init__(self, config):
        self.config = config
        self.global_step = tf.Variable(0, name="global_step", trainable=False)  # Epoch
        self.define_weights()

    def weight_variable(self, name, shape):
        if self.config.init == 1:
            #initial = tf.truncated_normal(shape, stddev = math.sqrt(1.0/shape[0]))
            return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        else: #He et al
            initial = tf.truncated_normal(shape, stddev= math.sqrt(2.0/shape[0]))
            return tf.Variable(initial, name=name)

    def bias_variable(self, name, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def define_weights(self):

        n_classes = 10
        # Store layers weight & bias
        self.weights = {
            # 3x3 conv, 3 input, 64 outputs
            'wc1': self.weight_variable('wc1',[3, 3, 3, 64]),
            # 3x3 conv, 64 inputs, 128 outputs
            'wc2': self.weight_variable('wc2',[3, 3, 64, 128]),
            # 3x3 conv, 128 inputs, 256 outputs
            'wc3': self.weight_variable('wc3',[3, 3, 128, 256]),
            # 3x3 conv, 256 inputs, 256 outputs
            'wc4': self.weight_variable('wc4',[3, 3, 256, 256]),

            # fully connected, 4*4*256 inputs, 1024 outputs
            'wd1': self.weight_variable('wd1',[4 * 4 * 256, 1024]),
            # fully connected, 1024 inputs, 1024 outputs
            'wd2': self.weight_variable('wd2',[1024,1024]),
            # 1024 inputs, 10 outputs (class prediction)
            'out': self.weight_variable('out',[1024, n_classes])
        }

        self.biases = {
            'bc1': self.bias_variable('bc1',[64]),
            'bc2': self.bias_variable('bc2',[128]),
            'bc3': self.bias_variable('bc3',[256]),
            'bc4': self.bias_variable('bc4',[256]),
            'bd1': self.bias_variable('bd1',[1024]),
            'bd2': self.bias_variable('bd2',[1024]),
            'bout': self.bias_variable('bout',[n_classes])
        }

    def conv2d(self,x, W, b, name, phase, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(input=x, filter= W, strides=[1, strides, strides, 1] , padding="SAME", name=name)
        x = tf.nn.bias_add(x, b)
        #x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase, scope=name+'_BN')

        x = tf.nn.relu(x)
        return x

    def maxpool2d(self, x, name, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

    def predict(self, x, keep_prob, phase):


        # Reshape input picture
        #x = tf.reshape(x, shape=[-1, 3, 32, 32])
        #x = tf.transpose(x, [0, 2, 3, 1])
        # tf.summary.image('images', x)
        self.x = x
        # Convolution Layer
        conv1 = self.conv2d(x, self.weights['wc1'], self.biases['bc1'], 'conv1', phase)
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, name='max1', k=2)
        conv1 = tf.nn.dropout(conv1, keep_prob)

        # Convolution Layer
        conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 'conv2', phase)
        conv2 = self.maxpool2d(conv2, name='max2', k=2)
        conv2 = tf.nn.dropout(conv2, keep_prob)

        # Convolution Layer
        conv3 = self.conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 'conv3', phase)
        self.conv3op = conv3
        conv3 = tf.nn.dropout(conv3, keep_prob)

        # Convolution Layer
        conv4 = self.conv2d(conv3, self.weights['wc4'], self.biases['bc4'], 'conv4', phase)
        conv4 = self.maxpool2d(conv4, name='max4', k=2)
        conv4 = tf.nn.dropout(conv4, keep_prob)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv4, [-1, self.weights['wd1'].get_shape().as_list()[0]], name='reshape')
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'], name='fc1')
        #fc1 = tf.contrib.layers.batch_norm(fc1, center=True, scale=True, is_training=phase, scope='fc1_BN')
        fc1 = tf.nn.relu(fc1, name='fcrelu1')
        fc1 = tf.nn.dropout(fc1, keep_prob)

        fc2 = tf.add(tf.matmul(fc1, self.weights['wd2']), self.biases['bd2'], name='fc2')
        fc2 = tf.contrib.layers.batch_norm(fc2, center=True, scale=True, is_training=phase, scope='fc2_BN')
        fc2 = tf.nn.relu(fc2, name='fcrelu2')
        fc2 = tf.nn.dropout(fc2, keep_prob)

        # Output, class prediction
        out = tf.add(tf.matmul(fc2, self.weights['out']), self.biases['bout'], name='out')
        return out

    def loss(self, predictions, labels):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels))
        tf.add_to_collection('Loss', cost)
        loss = tf.add_n(tf.get_collection('Loss'))
        return cost

    def training(self, loss, optimizer):
        train_op = optimizer.minimize(loss)
        return train_op

    def get_n_correct_predictions(self, predictions, labels):
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(tf.nn.softmax(predictions), 1), tf.argmax(labels, 1))
        return tf.reduce_sum(tf.cast(correct_pred,tf.int32))
