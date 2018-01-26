import numpy as np


class Cost:

    def __call__(self, prediction, target):
        raise NotImplementedError

    def delta(self, prediction, target):
        raise NotImplementedError


class sq(Cost):
    """
    Squared error loss
    """

    def __call__(self, prediction, target):
        y = np.zeros(prediction.shape)
        y[np.arange(y.shape[0]),target] = 1
        return (prediction - y) ** 2 / 2

    def delta(self, prediction, target):
        y = np.zeros(prediction.shape)
        y[np.arange(y.shape[0]),target] = 1
        return prediction - y


class ce(Cost):
    """
    Cross Entropy with clipping for stabilization
    """

    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def __call__(self, prediction, target):
        y = np.zeros(prediction.shape)
        y[np.arange(y.shape[0]),target] = 1
        target = y
        clipped = np.clip(prediction, self.epsilon, 1 - self.epsilon)
        cost = target * np.log(clipped) + (1 - target) * np.log(1 - clipped)
        return -cost

    def delta(self, prediction, target):
        y = np.zeros(prediction.shape)
        y[np.arange(y.shape[0]),target] = 1
        target = y
        denominator = np.maximum(prediction - prediction ** 2, self.epsilon)
        delta = (prediction - target) / denominator
        assert delta.shape == target.shape == prediction.shape
        return delta
