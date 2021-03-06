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
       calculate hodrick sum of each matrix column
       if forecast_horizon = 1 then the hodrick sum is the initial series

       :param X: independent regression variable
       :param forecast_horizon: forecast horizon
       """
    # X = HodrickSum(X)/sigma2_t
    # Initialize X(t)
    ret = np.cumsum(data, axis=0)
    ret[forecast_horizon:] = ret[forecast_horizon:] - ret[:-forecast_horizon]
    return ret[forecast_horizon - 1:]


def scale_matrix(x):
    """
    calculates expected value (here sample average) of X'X
    :param x: matrix with constant in first column
    :return: returns scale parameter
    """
    # Matrix multiplication/division solution
    E_x = np.dot(np.transpose(x), x)
    E_x = np.divide(E_x, x.shape[0])

    return E_x


if __name__ == "__main__":
    """
    Test functions if they correct sum up and return correct dimensions
    """
    # test vector
    x = [1,2,3,4,5,6,7]
    # check rolling sum forecast horizon 1
    print(rolling_sum(x,1))
    # test matrix
    y = np.matrix([[1, 1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6, 7]])
    # check hodrick sum forecast horizon 1
    print(hodrick_sum(x,1))
    print(hodrick_sum(y.T, 1))

    # check rolling sum forecast horizon > 1
    print(rolling_sum(x, 3))
    # check hodrick sum forecast horizon > 1
    print("multivariate")
    print(hodrick_sum(y.T, 3))


    # Check expected value calculation of X'X
    #y = np.matrix([[1,2], [1,2], [1,2], [1,2], [1,2]])
    print(scale_matrix(y.T))
    print(scale_matrix_old(y.T))






