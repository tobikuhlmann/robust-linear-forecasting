__author__ = 'Simon Walther'

import numpy as np
from scipy.stats import f, skew, kurtosis
import matplotlib.pyplot as plt


class OLS:
    def __init__(self, X, y, neweyWestLags=None, sampleSplit=0.7):
        """
        Represents an influence between a set of influencers and an influenced variable.

        :param X: numpy matrix. TxN. T number of time indices, N number of influencers
        :param y: numpy vector. Influenced variables.
        :param neweyWestLags: int. Number of lags used for the Newey-West correction. If None, 4 * (T/100)^(2/9) will be used.
        :param nearestNeighbor: int. Defines, which nearest neighbor will be used in the bandwidth selection for the
                                kernel regression.
        :param kernel: string. Kernel that is used in the kernel regression. Can be: "gaussian"
        :param kernelConfidence: float. Confidence level for the kernel regression confidence intervals.
        :param sampleSplit: float. Portion of the time series that is used for in-sample estimation. The first portion of
                            the inputs will be used for the in-sample series. The remaining part is used as out-of-sample
                            part. Daily model reestimation will be performed.
        """
        usedX = X.copy()
        if len(usedX.shape) == 1:
            usedX = usedX.reshape((usedX.shape[0], 1))
        self.X = np.concatenate((np.ones((usedX.shape[0], 1)), usedX), axis=1)
        self.T = usedX.shape[0]
        self.N = usedX.shape[1] + 1
        self.y = y.reshape((self.T, 1))
        if neweyWestLags == None:
            neweyWestLags = int(round(4 * ((self.T / 100.0) ** (2.0 / 9.0)), 0))
        self.neweyWestLags = neweyWestLags
        self.sampleSplit = sampleSplit

    def fit(self):
        """
        Performs fitting of the models. After fitting the results are stored in the Influence object and can be printed by
        using the printResults version.
        """
        # In-sample linear regression
        self.beta = self._linreg(self.X, self.y)
        self.linresid_in = self.y - np.dot(self.X, self.beta)
        self.beta_cov = self._calcBetaCov(self.X, self.linresid_in)
        self.beta_se = np.ones(self.beta.shape)
        for i in range(0, self.beta.shape[0]):
            self.beta_se[i, 0] = (self.beta_cov[i, i] / self.T) ** 0.5
        self.lin_in_rmse = np.mean(self.linresid_in ** 2) ** 0.5
        self.lin_in_r2 = 1.0 - (self.lin_in_rmse ** 2) / np.var(self.y)
        self.lin_in_residskew = skew(self.linresid_in)
        self.lin_in_residexcesskurt = kurtosis(self.linresid_in)
        fstat = np.var(self.y, ddof=1) / (np.sum(self.linresid_in ** 2) / (self.T - 1))
        self.lin_in_fp = f.cdf(fstat, self.T - 1, self.T - 1)
        if self.lin_in_fp > 0.5:
            self.lin_in_fp = 1 - self.lin_in_fp

        # Out-of-sample linear regression
        separationIndex = int(round(self.T * self.sampleSplit))
        out_linresid = np.zeros((self.y.shape[0] - separationIndex,))
        for j in range(separationIndex, self.y.shape[0]):
            beta = self._linreg(self.X[:j, :], self.y[:j])
            out_linresid[j - separationIndex] = self.y[j] - np.dot(self.X[j:(j + 1), :], beta)
        self.lin_out_rmse = np.mean(out_linresid ** 2) ** 0.5
        self.lin_out_r2 = 1.0 - (self.lin_out_rmse ** 2) / np.var(self.y[separationIndex:])
        self.lin_out_residskew = skew(out_linresid)
        self.lin_out_residexcesskurt = kurtosis(out_linresid)
        fstat = np.var(self.y[separationIndex:], ddof=1) / (np.sum(out_linresid ** 2) / (out_linresid.shape[0] - 1))
        self.lin_out_fp = f.cdf(fstat, out_linresid.shape[0] - 1, out_linresid.shape[0] - 1)
        self.linresid_out = out_linresid
        if self.lin_out_fp > 0.5:
            self.lin_out_fp = 1 - self.lin_out_fp


    def printResults(self):
        """
        Prints out several information items for the influence. The influence must have been fitted priorly. Some
        information, like model fit charts or kernel regression, is only performed if the X variable of the influence
        is 1D.
        """

        # Print Results
        print("OLS Evaluation")
        print("-------------------------------------------------------------------------------------------------------")
        print("OLS betas: {}".format(self.beta))
        print("OLS betas standard errors: {}".format(self.beta_se))
        print("In sample R_squared: {}".format(round(self.lin_in_r2, 4)))
        print("Out of sample R_squared: {}".format(round(self.lin_out_r2, 4)))
        print("-------------------------------------------------------------------------------------------------------")


    def _linreg(self, X, y):
        beta = np.dot(np.dot(np.linalg.solve(np.dot(X.T, X), np.identity(X.shape[1])), X.T), y)
        return beta

    def _calcSigma(self, X, resid, j):
        h = X.T * np.repeat(resid.T, X.shape[1], axis=0)
        if j == 0:
            sigma = np.dot(h, h.T) / float(X.shape[0])
        else:
            sigma = np.dot(h[:, j:], h[:, :-j].T) / float(X.shape[0])
        return sigma

    def _calcBetaCov(self, X, resid):
        HAC = self._calcSigma(X, resid, 0)
        for j in range(1, self.neweyWestLags):
            sigma_j = self._calcSigma(X, resid, j)
            HAC += (1.0 - float(j) / float(self.neweyWestLags + 1)) * (sigma_j + sigma_j.T)
        Q = np.dot(X.T, X) / self.T
        Q_inv = np.linalg.solve(Q, np.identity(self.N))
        beta_cov = np.dot(np.dot(Q_inv, HAC), Q_inv)
        return beta_cov



if __name__ == "__main__":
    # x = np.abs(np.random.normal(size = 1000))
    x = np.abs(np.random.uniform(0, 5, size=1000))
    # x = np.linspace(0, 5, 1000)
    y = 0.7 * x ** 2 + np.random.normal(loc=0, scale=0.2, size=1000)
    inf = OLS(x, y)
    inf.fit()
    inf.printResults()
