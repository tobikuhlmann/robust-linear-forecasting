import numpy as np
from scipy.stats import f, skew, kurtosis
import matplotlib.pyplot as plt
from tabulate import tabulate

class Influence:


    def __init__(self, X, y, neweyWestLags = None, kernel = "gaussian", nearestNeighbor = 10, kernelConfidence = 0.9, sampleSplit = 0.7):
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
        self.X = np.concatenate((np.ones((usedX.shape[0], 1)), usedX), axis = 1)
        self.T = usedX.shape[0]
        self.N = usedX.shape[1] + 1
        self.y = y.reshape((self.T, 1))
        if neweyWestLags == None:
            neweyWestLags = int(round(4 * ((self.T / 100.0) ** (2.0 / 9.0)), 0))
        self.neweyWestLags = neweyWestLags
        self.kernel = kernel
        self.kernelConfidence = kernelConfidence
        self.sampleSplit = sampleSplit
        self.nearestNeighbor = nearestNeighbor


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
        fstat = np.var(self.y, ddof = 1) / (np.sum(self.linresid_in ** 2) / (self.T - 1))
        self.lin_in_fp = f.cdf(fstat, self.T - 1, self.T - 1)
        if self.lin_in_fp > 0.5:
            self.lin_in_fp = 1 - self.lin_in_fp

        # Out-of-sample linear regression
        separationIndex = int(round(self.T * self.sampleSplit))
        out_linresid = np.zeros((self.y.shape[0] - separationIndex,))
        for j in range(separationIndex, self.y.shape[0]):
            beta = self._linreg(self.X[:j, :], self.y[:j])
            out_linresid[j - separationIndex] = self.y[j] - np.dot(self.X[j:(j+1), :], beta)
        self.lin_out_rmse = np.mean(out_linresid ** 2) ** 0.5
        self.lin_out_r2 = 1.0 - (self.lin_out_rmse ** 2) / np.var(self.y[separationIndex:])
        self.lin_out_residskew = skew(out_linresid)
        self.lin_out_residexcesskurt = kurtosis(out_linresid)
        fstat = np.var(self.y[separationIndex:], ddof=1) / (np.sum(out_linresid ** 2) / (out_linresid.shape[0] - 1))
        self.lin_out_fp = f.cdf(fstat, out_linresid.shape[0] - 1, out_linresid.shape[0] - 1)
        self.linresid_out = out_linresid
        if self.lin_out_fp > 0.5:
            self.lin_out_fp = 1 - self.lin_out_fp

        # In-sample kernel regression
        if self.N > 2:
            self.hasKernelReg = False
            return
        self.hasKernelReg = True
        kernelFit = self._doKernelRegression(self.X[:,1], self.y, self.X[:,1])
        self.kernelresid_in = self.y[:,0] - kernelFit
        self.kernel_in_rmse = np.mean(self.kernelresid_in ** 2) ** 0.5
        self.kernel_in_r2 = 1.0 - (self.kernel_in_rmse ** 2) / np.var(self.y)
        self.kernel_in_residskew = skew(self.kernelresid_in)
        self.kernel_in_residexcesskurt = kurtosis(self.kernelresid_in)
        fstat = np.var(self.y, ddof=1) / (np.sum(self.kernelresid_in ** 2) / (self.T - 1))
        self.kernel_in_fp = f.cdf(fstat, self.T - 1, self.T - 1)
        if self.kernel_in_fp > 0.5:
            self.kernel_in_fp = 1 - self.kernel_in_fp

        # Out-of-sample kernel regression
        out_kernelresid = np.zeros((self.y.shape[0] - separationIndex,))
        for j in range(separationIndex, self.y.shape[0]):
            fit = self._doKernelRegression(self.X[:j,1], self.y[:j], self.X[j:(j+1),1])
            out_kernelresid[j - separationIndex] = self.y[j,0] - fit[0]
        self.kernel_out_rmse = np.mean(out_kernelresid ** 2) ** 0.5
        self.kernel_out_r2 = 1.0 - (self.kernel_out_rmse ** 2) / np.var(self.y[separationIndex:])
        self.kernel_out_residskew = skew(out_kernelresid)
        self.kernel_out_residexcesskurt = kurtosis(out_kernelresid)
        fstat = np.var(self.y[separationIndex:], ddof=1) / (np.sum(out_kernelresid ** 2) / (out_kernelresid.shape[0] - 1))
        self.kernel_out_fp = f.cdf(fstat, out_kernelresid.shape[0] - 1, out_kernelresid.shape[0] - 1)
        self.kernelresid_out = out_kernelresid
        if self.kernel_out_fp > 0.5:
            self.kernel_out_fp = 1 - self.kernel_out_fp


    def printResults(self, X_name = [""], y_name = ""):
        """
        Prints out several information items for the influence. The influence must have been fitted priorly. Some
        information, like model fit charts or kernel regression, is only performed if the X variable of the influence
        is 1D.
        """
        if self.hasKernelReg:
            regX = np.linspace(np.min(self.X[:,1]), np.max(self.X[:,1]), 200)
            linreg_fit = self.beta[0] + self.beta[1] * regX
            kernel_fit = self._doKernelRegression(self.X[:,1], self.y, regX)
            kernel_lower, kernel_upper = self._calcKernelConfidenceIntervals(self.X[:,1], self.y, regX)
            plt.scatter(self.X[:,1], self.y, color = "black", label = "observations")
            plt.plot(regX, linreg_fit, color = "red", lw = 2, label = "linear fit")
            plt.plot(regX, kernel_fit, color = "blue", lw = 2, label = "kernel fit")
            plt.plot(regX, kernel_lower, color = "cyan", label = "%s%% confidence bound" % (int(round(self.kernelConfidence * 100))))
            plt.plot(regX, kernel_upper, color = "cyan", label = "%s%% confidence bound" % (int(round(self.kernelConfidence * 100))))
            plt.title("Observations and Fits")
            plt.xlabel(X_name[0])
            plt.ylabel(y_name)
            plt.legend()
            plt.show()

        resid_in = np.linspace(np.min(self.linresid_in), np.max(self.linresid_in), 100)
        resid_out = np.linspace(np.min(self.linresid_out), np.max(self.linresid_out), 100)
        if self.hasKernelReg:
            resid_in = np.linspace(min(np.min(self.linresid_in), np.min(self.kernelresid_in)),
                                   max(np.max(self.linresid_in), np.max(self.kernelresid_in)), 100)
            resid_out = np.linspace(min(np.min(self.linresid_out), np.min(self.kernelresid_out)),
                                    max(np.max(self.linresid_out), np.max(self.kernelresid_out)), 100)
        dens_lin_in = self._calcKernelDensity(self.linresid_in[:,0], resid_in)
        dens_lin_out = self._calcKernelDensity(self.linresid_out, resid_out)
        if self.hasKernelReg:
            dens_kernel_in = self._calcKernelDensity(self.kernelresid_in, resid_in)
            dens_kernel_out = self._calcKernelDensity(self.kernelresid_out, resid_out)
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(resid_in, dens_lin_in, color = "red", label = "linear fit")
        if self.hasKernelReg:
            ax[0].plot(resid_in, dens_kernel_in, color = "blue", label = "kernel fit")
        ax[0].set_title("In-sample residual densities per model")
        ax[0].set_xlabel("Residual")
        ax[0].set_ylabel("Probability Density")
        ax[0].legend()
        ax[1].plot(resid_out, dens_lin_out, color="red", label="linear fit")
        if self.hasKernelReg:
            ax[1].plot(resid_out, dens_kernel_out, color="blue", label="kernel fit")
        ax[1].set_title("Out-of-sample residual densities per model")
        ax[1].set_xlabel("Residual")
        ax[1].set_ylabel("Probability Density")
        ax[1].legend()
        plt.show()

        tbl = [["alpha", "%s (%s)" % (round(self.beta[0,0], 4), round(self.beta_se[0,0], 4)), "-"]]
        for i in range(0, self.beta.shape[0] - 1):
            nam = X_name[0]
            if nam == "":
                nam = "beta%s" % (i + 1)
            tbl.append([nam, "%s (%s)" % (round(self.beta[i + 1,0], 4), round(self.beta_se[i + 1,0], 4)), "-"])
        tbl.append(["", "", ""])
        if self.hasKernelReg:
            tbl.append(["In-sample R^2", round(self.lin_in_r2, 4), round(self.kernel_in_r2, 4)])
            tbl.append(["In-sample RMSE", round(self.lin_in_rmse, 4), round(self.kernel_in_rmse, 4)])
            tbl.append(["In-sample F-test p-value", round(self.lin_in_fp, 4), round(self.kernel_in_fp, 4)])
            tbl.append(["In-sample Residual Skew", round(self.lin_in_residskew[0], 4), round(self.kernel_in_residskew, 4)])
            tbl.append(["In-sample Residual Excess Kurtosis", round(self.lin_in_residexcesskurt[0], 4), round(self.kernel_in_residexcesskurt, 4)])
            tbl.append(["", "", ""])
            tbl.append(["Out-of-sample R^2", round(self.lin_out_r2, 4), round(self.kernel_out_r2, 4)])
            tbl.append(["Out-of-sample RMSE", round(self.lin_out_rmse, 4), round(self.kernel_out_rmse, 4)])
            tbl.append(["Out-of-sample F-test p-value", round(self.lin_out_fp, 4), round(self.kernel_out_fp, 4)])
            tbl.append(["Out-of-sample Residual Skew", round(self.lin_out_residskew, 4), round(self.kernel_out_residskew, 4)])
            tbl.append(["Out-of-sample Residual Excess Kurtosis", round(self.lin_out_residexcesskurt, 4), round(self.kernel_out_residexcesskurt, 4)])
        else:
            tbl.append(["In-sample R^2", round(self.lin_in_r2, 4), "-"])
            tbl.append(["In-sample RMSE", round(self.lin_in_rmse, 4), "-"])
            tbl.append(["In-sample F-test p-value", round(self.lin_in_fp, 4), "-"])
            tbl.append(["In-sample Residual Skew", round(self.lin_in_residskew, 4), "-"])
            tbl.append(["In-sample Residual Excess Kurtosis", round(self.lin_in_residexcesskurt, 4), "-"])
            tbl.append(["", "", ""])
            tbl.append(["Out-of-sample R^2", round(self.lin_out_r2, 4), "-"])
            tbl.append(["Out-of-sample RMSE", round(self.lin_out_rmse, 4), "-"])
            tbl.append(["Out-of-sample F-test p-value", round(self.lin_out_fp, 4), "-"])
            tbl.append(["Out-of-sample Residual Skew", round(self.lin_out_residskew, 4), "-"])
            tbl.append(["Out-of-sample Residual Excess Kurtosis", round(self.lin_out_residexcesskurt, 4), "-"])
        print(tabulate(tbl, headers = ["", "Linear model", "Non-parametrix (kernel)"], tablefmt = "orgtbl"))


    def _linreg(self, X, y):
        beta = np.dot(np.dot(np.linalg.solve(np.dot(X.T, X), np.identity(X.shape[1])), X.T), y)
        return beta


    def _calcSigma(self, X, resid, j):
        h = X.T * np.repeat(resid.T, X.shape[1], axis = 0)
        if j == 0:
            sigma = np.dot(h, h.T) / float(X.shape[0])
        else:
            sigma = np.dot(h[:,j:], h[:,:-j].T) / float(X.shape[0])
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


    def _doKernelRegression(self, X_obs, y_obs, X_req):
        weights, _ = self._doBandwidthAndKernelWeight(X_obs, X_req)
        weightedObs = weights * np.repeat(y_obs.reshape((y_obs.shape[0], 1)), weights.shape[1], axis = 1)
        y_req = np.sum(weightedObs, axis = 0) / np.sum(weights, axis = 0)
        return y_req


    def _doBandwidthAndKernelWeight(self, X_obs, X_req):
        """diff = np.repeat(X_obs.reshape((X_obs.shape[0], 1)), X_req.shape[0], axis = 1)
        req_inflated = np.repeat(X_req.reshape((1, X_req.shape[0])), X_obs.shape[0], axis = 0)
        diff = diff - req_inflated
        h = np.sort(np.abs(diff), axis = 0)
        h = h[self.nearestNeighbor:(self.nearestNeighbor + 1), :]
        diff = diff / np.repeat(h, diff.shape[0], axis = 0)
        if self.kernel == "gaussian":
            w = np.exp(- ((diff ** 2) / 2.0))
        return w, h"""
        T = X_obs.shape[0]
        h = np.repeat(X_obs.reshape((T, 1)), T, axis = 1) - np.repeat(X_obs.reshape((1, T)), T, axis = 0)
        h = np.sort(np.abs(h), axis = 0)
        h = h[self.nearestNeighbor, :]
        diff = np.repeat(X_obs.reshape((T, 1)), X_req.shape[0], axis = 1) - np.repeat(X_req.reshape((1, X_req.shape[0])), T, axis = 0)
        diff = diff / np.repeat(h.reshape((T, 1)), X_req.shape[0], axis = 1)
        if self.kernel == "gaussian":
            w = np.exp(- ((diff ** 2) / 2.0))
        w = w / np.repeat(h.reshape((T, 1)), X_req.shape[0], axis = 1)
        return w, h

    def _calcKernelDensity(self, X_obs, X_req):
        #weights, bandwidth = self._doBandwidthAndKernelWeight(X_obs, X_req)
        #density = np.sum(weights, axis = 0) / (X_obs.shape[0] * bandwidth[0])
        weights, _ = self._doBandwidthAndKernelWeight(X_obs, X_req)
        density = np.sum(weights, axis = 0) / (X_obs.shape[0])
        sortingIndices = np.argsort(X_req)
        sorted_X_req = X_req[sortingIndices]
        sorted_density = density[sortingIndices]
        integrated_density = np.sum(((sorted_density[1:] + sorted_density[:-1]) / 2.0) * (sorted_X_req[1:] - sorted_X_req[:1]))
        density /= integrated_density
        return density

    def _calcKernelConfidenceIntervals(self, X_obs, y_obs, X_req):
        fullRange = np.max(y_obs) - np.min(y_obs)
        y_req = np.linspace(np.min(y_obs) - 0.5 * fullRange, np.max(y_obs) + 0.5 * fullRange, 5000).reshape((5000, 1))
        weights_x, _ = self._doBandwidthAndKernelWeight(X_obs, X_req)
        lowerConf = np.zeros(X_req.shape)
        upperConf = np.zeros(X_req.shape)
        for i in range(0, X_req.shape[0]):
            #weights_y, bandwidth_y = self._doBandwidthAndKernelWeight(y_obs[:,0], y_req[:,0])
            weights_y, _ = self._doBandwidthAndKernelWeight(y_obs[:,0], y_req[:,0])
            relevant_x = weights_x[:, i:(i+1)]
            #joint = np.sum(weights_y * np.repeat(relevant_x, y_req.shape[0], axis = 1), axis = 0) / bandwidth_y[0]
            joint = np.sum(weights_y * np.repeat(relevant_x, y_req.shape[0], axis = 1), axis = 0)
            cond_dens = joint / np.sum(relevant_x)
            prob = np.cumsum(((cond_dens[1:] + cond_dens[:-1]) / 2.0) * (y_req[1:,0] - y_req[:1,0]))
            prob /= prob[-1]
            confIndex = np.where(prob < ((1.0 - self.kernelConfidence) / 2.0))
            lowerConf[i] = y_req[confIndex[0][-1]]
            confIndex = np.where(prob > 1.0 - ((1.0 - self.kernelConfidence) / 2.0))
            upperConf[i] = y_req[confIndex[0][0]]
        return lowerConf, upperConf



if __name__ == "__main__":
    #x = np.abs(np.random.normal(size = 1000))
    x = np.abs(np.random.uniform(0, 5, size = 1000))
    #x = np.linspace(0, 5, 1000)
    y = 0.7 * x**2 + np.random.normal(loc = 0, scale = 0.2, size = 1000)
    inf = Influence(x, y)
    inf.fit()
    inf.printResults(X_name = ["X"], y_name = "y")