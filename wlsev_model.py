__author__ = 'Tobias Kuhlmann'

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from helper_functions import rolling_sum


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

        self.betas = None
        self.std_errors = None
        self.t_stats = None

        self.mse_benchmark = None
        self.mse_wlsev = None
        self.oos_r_squared = None
        self.rmse_in_sample = None
        self.var_in_sample = None
        self.in_sample_r_squared = None

        #print('WLS-EV Regression Object initialized!')

    def fit(self, summary = False):
        """
        Estimate wls-ev with correct function, depending on forecast horizon if returns are overlapping
        """
        if self.forecast_horizon == 1:
            # non-overlapping returns
            # Regress Y = r_(t+1)/sigma2 on X=X_t/sigma2

            # Get volatility from var through square root and delete last row of series to adjust dimensionality
            est_var_dim_adj = self.est_var ** 0.5
            # X = X_t/sigma2_t, delete last row to adjust for dimensionality
            X = self.X
            # add OLS constant
            X = sm.add_constant(X)
            # divide by variance
            X = X / est_var_dim_adj[:, None]
            # Y = r_(t+1)/sigma2_t
            y = self.y / est_var_dim_adj
            # Next, run ols regression to estimate the wlsev parameters
            wlsev_reg_model = sm.OLS(y, X)
            wlsev = wlsev_reg_model.fit()  # Fit the model

            # Error Statistics
            # Get robust standard errors Newey West (1987) with 6 lags
            robust_standard_errors = wlsev.get_robustcov_results(cov_type='HAC', maxlags=6)
            # betas
            self.betas = robust_standard_errors.params
            # standard errors
            self.std_errors = robust_standard_errors.bse
            # t-statistics
            self.t_stats = self.betas / self.std_errors

        else:
            # overlapping returns
            # Regress Y = r_(t+1)/sigma2 on X=HodrickSum

            # Get volatility from var through square root and delete last row of series to adjust dimensionality
            est_var_dim_adj = self.est_var[self.forecast_horizon - 1:] ** 0.5
            # X = HodrickSum(X)/sigma2_t
            # Initialize X(t)
            X = self.X[self.forecast_horizon - 1:]
            # Stack X and X(t-1) + X(t-2) + ... + X(t-(forecast horizon-1))
            for i in range(1, self.forecast_horizon):
                X = np.vstack((X, self.X[(self.forecast_horizon - (1 + i)):-i]))
            # Transpose X to get correct OLS dimensions
            X = np.transpose(X)
            X = np.sum(X, axis=1).reshape((X.shape[0], 1))
            # Calculate variances for scaling
            # Calculate Var(x_t_rolling): Variance of rolling sum
            Var_x_t_rolling = X.var()
            # Calculate Var(x_t): Variance of log return series x_t
            Var_x_t = self.X.var()
            # Continue X sum
            X = X / est_var_dim_adj.reshape((est_var_dim_adj.shape[0], 1))
            # add OLS constant
            X = sm.add_constant(X)
            # Y = r_(t+1)/sigma2_t
            y = self.y[self.forecast_horizon-1:] / est_var_dim_adj
            y = np.transpose(y)
            # Next, run ols regression to estimate the wlsev parameters
            wlsev_reg_model = sm.OLS(y, X)
            wlsev = wlsev_reg_model.fit()  # Fit the model

            # Error Statistics
            # Get robust standard errors Newey West (1987) with 6 lags
            robust_standard_errors = wlsev.get_robustcov_results(cov_type='HAC', maxlags=6)
            # robust_standard_errors.summary()

            # 2. Scale the resulting coefficients and standard errors
            # Scale Var_x_t_rolling/Var_x_t
            scale = Var_x_t_rolling / Var_x_t
            # Scale betas
            self.betas = scale * robust_standard_errors.params
            # Scale standard errors
            self.std_errors = scale * robust_standard_errors.bse
            # t-statistic
            self.t_stats = self.betas / self.std_errors

    def evaluate(self):
        """
        evaluate predictions based on estimated wls-ev mode
        """

        # Predict with wls-ev
        log_return_predict_wlsev = self.wls_ev_predict()
        # Predict with benchmark approach mean
        log_return_predict_benchmark = self.benchmark_predict()

        # define start index test set
        start_test_set = int(len(self.X) * 2 / 3)

        # Different methods for cummulativa vs day-ahead forecasting
        if self.forecast_horizon == 1:
            # Out of sample
            # Calculate MSE of wls-ev prediction, start at (test set index)+1, as prediction one period ahead
            self.mse_wlsev = np.mean((self.X[start_test_set + 1:] - log_return_predict_wlsev) ** 2)
            # Calculate MSE of benchmark prediction, start at (test set index)+1, as prediction one period ahead
            self.mse_benchmark = np.mean((self.X[start_test_set + 1:] - log_return_predict_benchmark) ** 2)

            # In sample
            lin_residuals_in_sample = self.y[:start_test_set-1] - (self.betas[0] + np.dot(self.X[:start_test_set-1], self.betas[1]))
            self.rmse_in_sample = np.mean(lin_residuals_in_sample ** 2) ** 0.5
            self.var_in_sample = np.var(self.y[:start_test_set-1])

        else:
            # calculate realized cummulative returns over forecast horizon sequences, start at (test set index)+1, as prediction one period ahead
            cum_rets_realized = rolling_sum(self.X[start_test_set + 1:], self.forecast_horizon)

            # Out of sample
            # Calculate MSE of wls-ev prediction, only where realized values are available
            self.mse_wlsev = np.mean((cum_rets_realized - log_return_predict_wlsev[:-self.forecast_horizon + 1]) ** 2)
            # Calculate MSE of benchmark prediction, only where realized values are available
            self.mse_benchmark = np.mean(
                (cum_rets_realized - log_return_predict_benchmark[:-self.forecast_horizon + 1]) ** 2)

            # In Sample
            lin_residuals_in_sample = rolling_sum(self.y[:start_test_set-1], self.forecast_horizon) - (self.betas[0] + np.dot(self.X[:start_test_set-self.forecast_horizon], self.betas[1]))
            self.rmse_in_sample = np.mean(lin_residuals_in_sample ** 2) ** 0.5
            self.var_in_sample = np.var(rolling_sum(self.y[:start_test_set-1], self.forecast_horizon))

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

        # Initialize result array with the length of test set -1 = length set - length training set
        log_return_predict_wlsev = np.empty(int(len(self.y)) - start_index_test - 1)

        # Loop through time series and calculate predictions with information available at t = i
        # Loop only to lengnth(set) -1, because we need realized values for our prediction for eval
        for i in range(start_index_test, len(self.y) - 1):
            # Initiate and Estimate model with information available at t = i
            wlsev_obj = Wlsev_model(self.X[:i], self.y[:i], self.est_var[:i], self.forecast_horizon)
            wlsev_obj.fit()
            betas, std_errors, t_stats = wlsev_obj.get_results()

            # Predict r_(t+1)
            if self.forecast_horizon == 1:
                # no constant for day ahead prediction
                log_return_predict_wlsev[i - start_index_test] = betas[0] + betas[1] * self.X[i]
            else:
                # with constant beta0, beta1 * last available value
                log_return_predict_wlsev[i - start_index_test] = betas[0] + betas[1] * self.X[i]

        return log_return_predict_wlsev

    def benchmark_predict(self):
        """
        Predict values based on mean of known values
        """

        # Get time series index for split train/test set
        start_index_test = int(len(self.y) * 2 / 3)

        # Initialize result array with the length of test set -1 = length set - length training set - 1
        log_return_predict_benchmark = np.empty(int(len(self.y)) - start_index_test - 1)

        # Loop through time series
        for i in range(start_index_test, len(self.y) - 1):
            # Predict r_(t+1)
            if self.forecast_horizon == 1:
                log_return_predict_benchmark[i - start_index_test] = np.mean(self.y[:i])
            else:
                # Calculate mean of rolling sum (=cummulative log returns)
                log_return_predict_benchmark[i - start_index_test] = np.mean(
                    rolling_sum(self.y[:i], self.forecast_horizon))

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