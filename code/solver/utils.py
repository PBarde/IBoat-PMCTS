import sys
sys.path.append("../model/")
import numpy as np


class Hist:
    """
    Definition of the histogram class.
    """
    #: Lower bound of histogram
    MIN_REWARD = 0

    #: Upper bound of histogram
    MAX_REWARD = 1.5

    #: Number of bins in histogram
    N_BINS = 12

    #: Values of the thresholds between bins (lower and upper included)
    THRESH = np.arange(MIN_REWARD, MAX_REWARD + (MAX_REWARD - MIN_REWARD) / N_BINS, (MAX_REWARD - MIN_REWARD) / N_BINS)

    #: mean value of each bin (mean between upper and lower threshold of each bin)
    MEANS = np.mean(np.stack((THRESH[:-1], THRESH[1:]), axis=0), axis=0)

    def __init__(self, init=[]):
        if len(init) == 0:
            self.h = np.zeros(Hist.N_BINS, dtype=int)
        else:
            self.h = np.array(init, dtype=int)

    def add(self, value):
        """
        Adds the value to the corresponding bin. If value is lower than lowest (resp. highest) threshold
        then the value is added to the first (resp. last) bin.
        :param int value: value to be added to the histogram

        """
        i = 0
        for x in Hist.THRESH[1:-1]:
            if value > x:
                i = i + 1
            else:
                break
        self.h[i] += 1

    def get_mean(self):
        """
        Computes the mean value of the histogram
        :return float mean: mean value
        """
        summed = sum(self.h)
        if summed == 0:
            return 0
        else:
            return np.dot(self.h, Hist.MEANS) / summed

    def is_empty(self):
        return all(value == 0 for value in self.h)
