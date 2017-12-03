import numpy as np

def rolling_sum(a, n) :
    # calculate rolling sum (= cummulative returns) of log return series a
    # @params
    #   a: log return series
    #   n: forecast horizon

    ret = np.cumsum(a, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]
