import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, model, num_classes, sigma):
        self.model = model
        self.num_classes = num_classes
        self.sigma = sigma

    def predict(self, x, n, alpha, batch_size):
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
                class returned by this method will equal g(x).
                This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
                for identifying the top category of a multinomial distribution.
                :param x: the input [channel x height x width]
                :param n: the number of Monte Carlo samples to use
                :param alpha: the failure probability
                :param batch_size: batch size to use when evaluating the base classifier
                :return: the predicted class, or ABSTAIN
                """
        self.model.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1+count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x, num, batch_size):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
                :param x: the input [channel x width x height]
                :param num: number of samples to collect
                :param batch_size:
                :return: an ndarray[int] of length num_classes containing the per-class counts
                """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num/batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device=x.device) * self.sigma
                predictions = self.model(batch+noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr, length):
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts