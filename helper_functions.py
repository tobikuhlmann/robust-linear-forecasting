__author__ = 'Tobias Kuhlmann'

import numpy as np


def rolling_sum(a, n):
    """
    calculate rolling sum (= cummulative returns) of log return series
    rolling sum length = forecast horizon forward looking

     :param a: log return series
     :param n: forecast horizon
    """

    ret = np.cumsum(a, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]

def hodrick_sum(data, forecast_horizon):
    """
        calculate hodrick sum

         :param X: independent regression variable
         :param forecast_horizon: forecast horizon
        """
    # X = HodrickSum(X)/sigma2_t
    # Initialize X(t)
    result = data[forecast_horizon - 1:]
    # Stack X and X(t-1) + X(t-2) + ... + X(t-(forecast horizon-1))
    for i in range(1, forecast_horizon):
        result = np.vstack((result, data[(forecast_horizon - (1 + i)):-i]))
    # Transpose X to get correct OLS dimensions
    result = np.transpose(result)
    result = np.sum(result, axis=1).reshape((result.shape[0], 1))
    return result



