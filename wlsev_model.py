__author__ = 'Tobias Kuhlmann'

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from helper_functions import rolling_sum, hodrick_sum, scale_matrix


class Wlsev_model(object):
    """
    This class implements the second wls-ev step:
    Estimate return regression WLS-EV beta using Johnson

    METHODS:
      'init'                              : Initialization of the object with the dependent and independent regression variables, and variance estimation
      'fit'                               : Estimate WLS-EV regression using Johnson (2016)
    """

    def __init__(self, X, y, volatility, forecast_horizon):
        '''
        Initialize object with es50 data
        '''

        self.X = X
        self.y = y
        self.est_var = volatility
        self.forecast_horizon = forecast_horizon

        # model
        self.betas = None
        self.std_errors = None
        self.t_stats = None

        # evaluation
        self.mse_benchmark = None
        self.mse_wlsev = None
        self.oos_r_squared = None
        self.rmse_in_sample = None
        self.var_in_sample = None
        self.in_sample_r_squared = None

        # plots
        self.log_return_predict_benchmark = None
        self.log_return_predict_wlsev = None

        #print('WLS-EV Regression Object initialized!')

    def fit(self):
        """
        Estimate wls-ev with correct function, depending on forecast horizon if returns are overlapping
        """
        # overlapping returns: Regress Y = r_(t+1) / sigma on X=HodrickSum / sigma

        # Get volatility from var through square root
        est_var_dim_adj = self.est_var[self.forecast_horizon - 1:] ** 0.5
        # X = HodrickSum(X)

        # Create X without sums: stack columns axis 1
        X = np.column_stack((np.ones(self.X.shape), self.X))

        # get multivariate hodrick sum (sum up each column seperate)
        X_hodrick_sum = hodrick_sum(data=X, forecast_horizon=self.forecast_horizon)

        # get matrix of scales with already weighted X
        scale_a = scale_matrix(X / self.est_var.reshape(self.est_var.shape[0], 1) ** 0.5)
        scale_b = scale_matrix(X_hodrick_sum/est_var_dim_adj.reshape(est_var_dim_adj.shape[0], 1))
        scale = np.dot(np.linalg.inv(scale_a), scale_b)

        # wlsev step: divide by vol
        X = np.divide(X_hodrick_sum, est_var_dim_adj.reshape(est_var_dim_adj.shape[0], 1))

        # Y = r_(t+1)/sigma
        y = np.divide(self.y[self.forecast_horizon-1:], est_var_dim_adj)
        y = np.transpose(y)

        # Next, run ols regression to estimate the wlsev parameters
        wlsev_reg_model = sm.OLS(y, X)
        wlsev = wlsev_reg_model.fit()  # Fit the model

        # Error Statistics
        # Get robust standard errors Newey West (1987) with 6 lags
        robust_standard_errors = wlsev.get_robustcov_results(cov_type='HAC', maxlags=6)
        # 2. Scale the resulting coefficients and standard errors
        # Scale parameter Var_x_t_rolling/Var_x_t, scale = 1 if forecast horizon = 1
        # Scale betas
        self.betas = np.dot(scale, robust_standard_errors.params)
        # Scale standard errors
        self.std_errors = np.dot(scale, robust_standard_errors.bse)
        # t-statistic
        self.t_stats = self.betas / self.std_errors

    def evaluate(self):
        """
        evaluate predictions based on estimated wls-ev mode
        """
        # define start index test set
        start_test_set = int(len(self.X) * 2 / 3)

        # Different methods for cummulative vs day-ahead forecasting
        if self.forecast_horizon == 1:
            # In sample
            lin_residuals_in_sample = self.y - (self.betas[0] + np.dot(self.X, self.betas[1]))
            self.rmse_in_sample = np.mean(lin_residuals_in_sample ** 2) ** 0.5
            self.var_in_sample = np.var(self.y)

            # Out of sample
            # Calculate MSE of wls-ev prediction
            self.mse_wlsev = np.mean((self.y[start_test_set:] - self.wls_ev_predict()) ** 2)
            # Calculate MSE of benchmark prediction
            self.mse_benchmark = np.mean((self.y[start_test_set:] - self.benchmark_predict()) ** 2)
        else:
            # In Sample with betas estimated on full time series
            lin_residuals_in_sample = rolling_sum(self.y, self.forecast_horizon) - (
                    self.betas[0] + np.dot(self.X[:-(self.forecast_horizon-1)], self.betas[1]))
            self.rmse_in_sample = np.mean(lin_residuals_in_sample ** 2) ** 0.5
            self.var_in_sample = np.var(rolling_sum(self.y, self.forecast_horizon))

            # Out of sample
            # calculate realized cummulative returns over forecast horizon sequences
            cum_rets_realized = rolling_sum(self.y[start_test_set:], self.forecast_horizon)
            # Calculate MSE of wls-ev prediction, only where realized values are available
            self.mse_wlsev = np.mean((cum_rets_realized - self.wls_ev_predict()[:-(self.forecast_horizon-1)]) ** 2)
            # Calculate MSE of benchmark prediction, only where realized values are available
            self.mse_benchmark = np.mean(
                (cum_rets_realized - self.benchmark_predict()[:-(self.forecast_horizon-1)]) ** 2)

        # Calculate out of sample r-squared
        self.oos_r_squared = 1 - (self.mse_wlsev / self.mse_benchmark)
        # Calculate in sample r-squared
        self.in_sample_r_squared = 1.0 - (self.rmse_in_sample ** 2) / self.var_in_sample

    def wls_ev_predict(self):
        """
        Predict values based on estimated wls-ev model
        """

        # Get time series index for split train/test set
        start_index_test = int(len(self.y) * 2 / 3)

        # Initialize result array with the length of test set = length set - length training set
        log_return_predict_wlsev = np.empty(int(len(self.y)) - start_index_test)

        # Loop through time series and calculate predictions with information available at t = i
        # python range is equivalent to [start_index_test, len(self.y))
        for i in range(start_index_test, int(len(self.y))):
            # Initiate and Estimate model with information available at t = i
            wlsev_obj_help = Wlsev_model(self.X[:i-1], self.y[:i-1], self.est_var[:i-1], self.forecast_horizon)
            wlsev_obj_help.fit()
            betas, std_errors, t_stats = wlsev_obj_help.get_results()

            # Predict initial regression r_t with r_t-1
            log_return_predict_wlsev[i - start_index_test] = betas[0] + np.dot(betas[1], self.X[i-1])
        self.log_return_predict_wlsev = log_return_predict_wlsev
        return log_return_predict_wlsev

    def benchmark_predict(self):
        """
        Predict values based on mean of known values
        """

        # Get time series index for split train/test set
        start_index_test = int(len(self.y) * 2 / 3)

        # Initialize result array with the length of test set = length set - length training set
        log_return_predict_benchmark = np.empty(int(len(self.y)) - start_index_test)

        # Loop through time series
        # python range is equivalent to [start_index_test, len(self.y))
        for i in range(start_index_test, int(len(self.y))):
            # Predict r_t with r_t-1
            # Calculate mean of rolling sum (=cummulative log returns)
            log_return_predict_benchmark[i - start_index_test] = np.mean(
                rolling_sum(self.y[:i-1], self.forecast_horizon))
        self.log_return_predict_benchmark = log_return_predict_benchmark
        return log_return_predict_benchmark

    def print_results(self):
        """
        Print wls-ev results
        """

        print("WLS-EV Estimation Results")
        print('Forecast Horizon: {}'.format(self.forecast_horizon))
        print("-------------------------------------------------------------------------------------------------------")
        print("betas: {}".format(np.around(self.betas,4)))
        print("robust bse standard errors: {}".format(np.around(self.std_errors,4)))
        print("t-stats: {}".format(np.around(self.t_stats,4)))
        print("In sample R_squared: {}".format(round(self.in_sample_r_squared,4)))
        print("Out of sample R_squared: {}".format(round(self.oos_r_squared,4)))
        print("-------------------------------------------------------------------------------------------------------")

    def get_results(self):
        """
        get wls-ev results
        """
        return self.betas, self.std_errors, self.t_stats

    def plot_results(self):
        """
        plot results

        """
        import matplotlib

        matplotlib.use
        import matplotlib.pyplot as plt

        matplotlib.style.use('ggplot')

        # benchmark prediction
        plt.plot(range(0, len(self.log_return_predict_benchmark)), self.log_return_predict_benchmark,
                 label='mean benchmark')
        # wlsev prediction
        plt.plot(range(0, len(self.log_return_predict_wlsev)), self.log_return_predict_wlsev,
                 label='wlsev')
        # realized returns
        plt.plot(range(0, len(rolling_sum(self.y[int(len(self.y) * 2 / 3):], self.forecast_horizon))),rolling_sum(self.y[int(len(self.y) * 2 / 3):], self.forecast_horizon),
                 label='realized')
        plt.title('WLS-EV Time Series')
        plt.xlabel('time')
        plt.ylabel('returns')
        plt.legend()
        plt.show()

    def plot_scatter(self):
        """
        plot scatter

        """
        import matplotlib

        matplotlib.use
        import matplotlib.pyplot as plt

        matplotlib.style.use('ggplot')

        # plot initial X and Y
        if self.forecast_horizon == 1:
            X = self.X[int(len(self.y) * 2 / 3):]
        else:
            X = self.X[int(len(self.y) * 2 / 3):-(self.forecast_horizon-1)]
        Y = rolling_sum(self.y[int(len(self.y) * 2 / 3):], self.forecast_horizon)
        plt.scatter(X, Y)
        plt.title('WLS-EV Scatter')
        plt.xlabel('X')
        plt.ylabel('Y')
        # plot wlsev prediction
        plt.plot(X, self.betas[0] + self.betas[1] * X,
                 label='wlsev')

        plt.legend()
        plt.show()

    def get_plot_data_wlsev(self):
        if self.forecast_horizon == 1:
            X = self.X[int(len(self.y) * 2 / 3):]
        else:
            X = self.X[int(len(self.y) * 2 / 3):-(self.forecast_horizon - 1)]
        Y = rolling_sum(self.y[int(len(self.y) * 2 / 3):], self.forecast_horizon)
        y_wlsev = self.betas[0] + self.betas[1] * X
        return X, Y, y_wlsev
