import numpy as np

class Activation:

    def __call__(self, incoming):
        raise NotImplementedError

    def delta(self, incoming, outgoing, above):
        """
        Compute the derivative of the cost with respect to the input of this
        activation function. Outgoing is what this function returned in the
        forward pass and above is the derivative of the cost with respect to
        the outgoing activation.
        """
        raise NotImplementedError

    def init_weights(self,shape):
        '''Return initialization values for weight that would be apt for the activation function'''


class identity(Activation):

    def __call__(self, incoming):
        return incoming

    def delta(self, incoming, outgoing, above):
        delta = np.ones(incoming.shape).astype(float)
        return delta * above

    def init_weights(self,shape):
        weights = np.identity(shape)
        bias = np.zeros(shape[1])
        return weights,bias

class sigmoid(Activation):

    def __call__(self, incoming):
        return 1 / (1 + np.exp(-incoming))

    def delta(self, incoming, outgoing, above):
        delta = outgoing * (1 - outgoing)
        assert delta.shape == above.shape == outgoing.shape
        return delta * above

    def init_weights(self,shape):
        np.random.seed(1234)
        weights = np.random.uniform(-4*np.sqrt(6.0/(shape[0]+shape[1])),4*np.sqrt(6.0/(shape[0]+shape[1])),shape)
        bias = np.zeros(shape[1])
        return weights,bias

class tanh(Activation):

    def __call__(self, incoming):
        return (-np.exp(-incoming) + np.exp(incoming)) / (np.exp(-incoming) + np.exp(incoming))

    def delta(self, incoming, outgoing, above):
        delta = 1 - outgoing * outgoing
        return delta*above

    def init_weights(self,shape):
        np.random.seed(1234)
        weights = np.random.uniform(-np.sqrt(6.0/(shape[0]+shape[1])),np.sqrt(6.0/(shape[0]+shape[1])),shape)
        bias = np.zeros(shape[1])
        return weights,bias


class relu(Activation):

    def __call__(self, incoming):
        return np.maximum(incoming, 0)

    def delta(self, incoming, outgoing, above):
        delta = np.greater(outgoing, 0).astype(float)
        return delta * above

    def init_weights(self,shape):
        np.random.seed(1234)
        weights = np.random.randn(shape[0],shape[1])*np.sqrt(2.0/shape[0])
        bias = np.zeros(shape[1])
        return weights, bias


class leaky_relu(Activation):

    def __call__(self, incoming):
        x = incoming
        x[np.where(incoming < 0)] = 0.01 * incoming[np.where(incoming < 0)]
        return x

    def delta(self, incoming, outgoing, above):
        delta = np.ones(outgoing.shape)
        delta[np.where(outgoing < 0)] = 0.01
        return delta * above

    def init_weights(self,shape):
        np.random.seed(1234)
        weights = np.random.randn(shape[0], shape[1])*np.sqrt(2.0/shape[0])
        bias = np.zeros(shape[1])
        return weights, bias



