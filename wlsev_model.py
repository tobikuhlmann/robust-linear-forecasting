
# Imports

# Import packages for econometric analysis
import numpy as np
from helper_functions import rolling_sum

# Import plotting library
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')



class Wlsev_model(object):

    # DESCRIPTION:
    #   This class implements the second wls-ev step:
    #   Estimate return regression WLS-EV beta using Johnson
    #
    # METHODS:
    #   'init'                              : Initialization of the object with the es50 log return and estimated volatility data
    #   'estimate_wls_ev'                   : Run correct model estimation based on forecast horizon (overlapping or non-overlapping)
    #   'estimate_wls_ev_non_overlapping'   : Estimate return regression beta WLS-EV using Johnson (2016) for non overlapping returns
    #   'estimate_wls_ev_overlapping'       : Estimate return regression beta WLS_EV using Johnson (2016) and Hodrick (1992) to account for overlapping returns
    #

    def __init__(self, log_returns, vol,  forecast_horizon):
        #
        # DESCRIPTION:
        #   Initialize object with es50 data
        #
        self.log_returns = log_returns
        self.est_var = vol
        self.forecast_horizon = forecast_horizon

        self.betas = None
        self.std_errors = None
        self.t_stats = None

        print('WLS-EV Regression Object initialized!')


    def estimate_wls_ev(self):
        #
        # DESCRIPTION:
        #   Estimate wls-ev with correct function, depending on forecast horizon if returns are overlapping
        #

        if self.forecast_horizon == 1:
            self.betas, self.std_errors, self.t_stats = self.estimate_wls_ev_non_overlapping()

        else:
            self.betas, self.std_errors, self.t_stats = self.estimate_wls_ev_overlapping()

        return self.betas, self.std_errors, self.t_stats


    def estimate_wls_ev_non_overlapping(self):
        #
        # DESCRIPTION:
        #   Estimate return regression beta with WLS-EV
        #

        # import libraries
        import pandas as pd
        import statsmodels.api as sm


        # Estimate Regression beta with WLS-EV
        # --------------------------------------------------------------------------------
        # Regress Y = r_(t+1)/sigma2 on X=X_t/sigma2

        # Get vol from var through square root and delete last row of series to adjust dimensionality
        est_var_dim_adj  = self.est_var[:-1]**0.5

        # X = X_t/sigma2_t, no constant since constant is already in X_t, delete last row to adjust for dimensionality
        X = self.log_returns[:-1]/est_var_dim_adj

        # Y = r_(t+1)/sigma2_t
        Y = self.log_returns[1:]/est_var_dim_adj

        # Next, run ols regression to estimate the wlsev parameters
        wlsev_reg_model = sm.OLS(Y, X)
        wlsev = wlsev_reg_model.fit()  # Fit the model


        # Error Statistics
        # --------------------------------------------------------------------------------
        # Get robust standard errors Newey West (1987) with 6 lags
        robust_standard_errors = wlsev.get_robustcov_results(cov_type='HAC', maxlags=6)

        # betas
        betas = robust_standard_errors.params
        # standard errors
        std_errors = robust_standard_errors.bse
        # t-statistics
        t_stats = betas / std_errors

        print("WLS-EV Estimation Results Non-Overlapping")
        print('Forecast Horizon: {}'.format(self.forecast_horizon))
        print("-------------------------------------------------------------------------------------------------------")
        print("betas: {}".format(betas))
        print("bse standard errors: {}".format(std_errors))
        print("t-stats: {}".format(t_stats))
        print("-------------------------------------------------------------------------------------------------------")

        return betas, std_errors, t_stats


    def estimate_wls_ev_overlapping(self):
        #
        # DESCRIPTION:
        #   Estimate return regression beta with WLS-EV
        #

        # import libraries
        import pandas as pd
        import statsmodels.api as sm


        # Estimate Regression beta with WLS-EV
        # --------------------------------------------------------------------------------
        # overlapping returns

        # TODO: 1. Estimate the non-overlapping regression r_(t+1) on rolling sum
        # Regress Y = r_(t+1)/sigma2 on X=HodrickSum

        # Get vol from var through square root and delete last row of series to adjust dimensionality
        est_var_dim_adj  = self.est_var[self.forecast_horizon-1:-1]**0.5

        # X = HodrickSum(X)/sigma2_t, no constant since constant is already in X
        # Initialize X(t) and divide by estimated sigma_t->t+1 for wls_ev
        X = self.log_returns[self.forecast_horizon - 1:-1]

        # Stack X and X(t-1) + X(t-2) + ... + X(t-(forecast horizon-1))
        for i in range(1, self.forecast_horizon):
            X = np.vstack((X, self.log_returns[(self.forecast_horizon - (1 + i)):-(1 + i)]))

        # Transpose X to get correct OLS dimensions
        X = np.transpose(X)
        X = np.sum(X, axis = 1).reshape((X.shape[0], 1))

        # Calculate variances for scaling
        # Calculate Var(x_t_rolling): Variance of rolling sum
        Var_x_t_rolling = X.var()
        # Calculate Var(x_t): Variance of log return series x_t
        Var_x_t = self.log_returns.var()

        # Continue X sum
        X = X / est_var_dim_adj.reshape((est_var_dim_adj.shape[0], 1))

        # add OLS constant
        X = sm.add_constant(X)

        # Y = r_(t+1)/sigma2_t
        Y = self.log_returns[self.forecast_horizon:] / est_var_dim_adj
        Y = np.transpose(Y)

        # Next, run ols regression to estimate the wlsev parameters
        wlsev_reg_model = sm.OLS(Y, X)
        wlsev = wlsev_reg_model.fit()  # Fit the model


        # Error Statistics
        # --------------------------------------------------------------------------------
        # Get robust standard errors Newey West (1987) with 6 lags
        robust_standard_errors = wlsev.get_robustcov_results(cov_type='HAC', maxlags=6)
        #robust_standard_errors.summary()

        # 2. Scale the resulting coefficients and standard errors
        # Scale Var_x_t_rolling/Var_x_t
        scale = Var_x_t_rolling / Var_x_t
        # Scale betas
        betas = scale * robust_standard_errors.params
        # Scale standard errors
        std_errors = scale * robust_standard_errors.bse
        # t-statistic
        t_stats = betas / std_errors

        # Print Results
        print("WLS-EV Estimation Results Overlapping")
        print('Forecast Horizon: {}'.format(self.forecast_horizon))
        print("-------------------------------------------------------------------------------------------------------")
        print("Scale: {}".format(scale))
        print("Scaled betas: {}".format(betas))
        print("Scaled bse standard errors: {}".format(std_errors))
        print("Scaled t-stats: {}".format(t_stats))
        print("-------------------------------------------------------------------------------------------------------")

        return betas, std_errors, t_stats


    def wls_ev_predict(self):
        #
        # DESCRIPTION:
        #   Predict values based on estimated wls-ev model
        #

        # Get time series index for split train/test set
        start_index_test = int(len(self.log_returns)*2/3)

        # Initialize result array with the length of test set -1 = length set - length training set
        log_return_predict_wlsev = np.empty(int(len(self.log_returns)) - start_index_test - 1)

        # Loop through time series and calculate predictions with information available at t = i
        # Loop only to lengnth(set) -1, because we need realized values for our prediction for eval
        for i in range(start_index_test, len(self.log_returns)-1):
            # Initiate and Estimate model with information available at t = i
            wlsev_obj = Wlsev_model(self.log_returns[:i], self.est_var[:i], self.forecast_horizon)
            betas, std_errors, t_stats = wlsev_obj.estimate_wls_ev()

            # Predict r_(t+1)
            if self.forecast_horizon ==1:
                # no constant for day ahead prediction
                log_return_predict_wlsev[i-start_index_test] = betas[0] * self.log_returns[i]
            else:
                # with constant beta0, beta1 * last available value
                log_return_predict_wlsev[i-start_index_test] = betas[0] + betas[1] * self.log_returns[i]

        return log_return_predict_wlsev


    def benchmark_predict(self):
        #
        # DESCRIPTION:
        #   Predict values based on mean of known values
        #

        # Get time series index for split train/test set
        start_index_test = int(len(self.log_returns)*2/3)

        # Initialize result array with the length of test set -1 = length set - length training set - 1
        log_return_predict_benchmark = np.empty(int(len(self.log_returns)) - start_index_test - 1)

        # Loop through time series
        for i in range(start_index_test, len(self.log_returns)-1):
            # Predict r_(t+1)
            if self.forecast_horizon == 1:
                log_return_predict_benchmark[i-start_index_test] = np.mean(self.log_returns[:i])
            else:
                # Calculate mean of rolling sum (=cummulative log returns)
                log_return_predict_benchmark[i - start_index_test] = np.mean(rolling_sum(self.log_returns[:i], self.forecast_horizon))

        return log_return_predict_benchmark


    def wls_ev_eval(self):
        #
        # DESCRIPTION:
        #   evaluate predictions based on estimated wls-ev mode
        #

        # Predict with wls-ev
        log_return_predict_wlsev = self.wls_ev_predict()

        # Predict with benchmark approach mean
        log_return_predict_benchmark = self.benchmark_predict()

        # define start index test set
        start_test_set = int(len(self.log_returns)*2/3)

        # Different methods for cummulativa vs day-ahead forecasting
        if self.forecast_horizon == 1:
            # Calculate MSE of wls-ev prediction, start at (test set index)+1, as prediction one period ahead
            mse_wlsev = np.mean((self.log_returns[start_test_set+1:] - log_return_predict_wlsev) ** 2)

            # Calculate MSE of benchmark prediction, start at (test set index)+1, as prediction one period ahead
            mse_benchmark = np.mean((self.log_returns[start_test_set+1:] - log_return_predict_benchmark) ** 2)
        else:
            # calculate realized cummulative returns over forecast horizon sequences, start at (test set index)+1, as prediction one period ahead
            cum_rets_realized = rolling_sum(self.log_returns[start_test_set + 1:], self.forecast_horizon)

            # Calculate MSE of wls-ev prediction, only where realized values are available
            mse_wlsev = np.mean((cum_rets_realized - log_return_predict_wlsev[:-self.forecast_horizon+1]) ** 2)

            # Calculate MSE of benchmark prediction, only where realized values are available
            mse_benchmark = np.mean((cum_rets_realized - log_return_predict_benchmark[:-self.forecast_horizon+1]) ** 2)


        # Calculate out of sample r-squared
        oos_r_squared = 1 - (mse_wlsev/mse_benchmark)

        # Print Results
        print("WLS-EV Evaluation")
        print('Forecast Horizon: {}'.format(self.forecast_horizon))
        print("-------------------------------------------------------------------------------------------------------")
        print("Benchmark model OOS MSE: {}".format(mse_benchmark))
        print("WLS-EV model OOS MSE: {}".format(mse_wlsev))
        print("Out of sample R_squared: {}".format(oos_r_squared))
        print("-------------------------------------------------------------------------------------------------------")

        return oos_r_squared
