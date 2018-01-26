import numpy as np
from scipy.special import expit
import sys
import time


class RBM(object):

    def __init__(self, n_vis, n_hid):
        self.n_vis = n_vis
        self.n_hid = n_hid

        self.weights = np.random.uniform(-1./10, 1./10, (n_vis, n_hid))
        self.vis_bias = np.random.uniform(-1./10, 1./10, (n_vis))
        self.hid_bias = np.random.uniform(-1./10, 1./10, (n_hid))

    def get_sample(self, probs):
        #samples = np.random.binomial(1, p=probs, size=probs.shape)
        #probs[samples > probs] = 1.
        samples = np.random.uniform(size=probs.shape)
        probs[samples < probs] = 1.
        np.floor(probs, probs)
        return probs

    def get_hidden(self, vis, sample=False):
        x = expit(np.dot(vis, self.weights) + self.hid_bias)
        if sample is True:
            return self.get_sample(x)
        return x

    def get_visible(self, hid, sample=False):
        x = expit(np.dot(hid, self.weights.T) + self.vis_bias)
        if sample is True:
            return self.get_sample(x)
        return x

    def gibbs_walk(self, n_steps, hid):
        t_hid = np.copy(hid)
        n_samples = hid.shape[0]
        sample = False

        for i in range(n_steps):
            t_vis = self.get_visible(t_hid, sample)
            if i == n_steps-1:
                sample = True
            t_hid = self.get_hidden(t_vis, sample)

        return t_vis, t_hid


class CDTrainer(object):

    def __init__(self, model, lr=1e-4, mr=0.9, wdecay=0.0002):
        self.model = model
        self.learning_rate = lr
        self.momentum_rate = mr
        self.weight_decay_rate = wdecay
        self.weightstep = np.zeros(model.weights.shape)

    def train(self, data, n_epochs, cdsteps=1, batch_size=100):
        model = self.model
        n_samples, n_dims = data.shape
        mse = []
        for epoch in range(n_epochs):
            epoch_start = time.clock()
            t_mse = 0
            for offset in xrange(0, n_samples, batch_size):
                #Do Gibbs Walk
                v_0 = data[offset:(offset+batch_size)]
                h_0 = model.get_hidden(v_0, sample=False)
                v_k, h_k = model.gibbs_walk(cdsteps, model.get_sample(h_0))
                #h_0 = model.get_hidden(v_0, sample=True)
                #v_k, h_k = model.gibbs_walk(cdsteps, h_0)

                #Weight Update
                momentum = self.momentum_rate*self.weightstep
                weight_update = (1./batch_size)*(np.matmul(v_0.T, h_0) - np.matmul(v_k.T, h_k))
                weight_decay = self.weight_decay_rate*model.weights
                self.weightstep = momentum + self.learning_rate*(weight_update - weight_decay)
                model.weights += self.weightstep

                #Bias Updates
                hid_bias_update = (1./batch_size)*(h_0.sum(axis=0) - h_k.sum(axis=0))
                model.hid_bias += self.learning_rate*hid_bias_update
                vis_bias_update = (1./batch_size)*(v_0.sum(axis=0) - v_k.sum(axis=0))
                model.vis_bias += self.learning_rate*vis_bias_update

                #Compute MSE
                t_mse += ((v_0 - v_k)**2).sum()/n_samples
            mse.append(t_mse)
            print "Done epoch %d: %f seconds, MSE=%f" % (epoch + 1, time.clock() - epoch_start, t_mse)
            sys.stdout.flush()
        np.save('mse', np.array(mse))

