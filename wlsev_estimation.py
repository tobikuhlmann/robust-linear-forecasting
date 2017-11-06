
# Imports

# Import packages for econometric analysis
import numpy as np
import pandas as pd
import statsmodels as sm

# Import plotting library
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')



class WlsEvEstimation(object):

    # DESCRIPTION:
    #   This class implements the second wls-ev step:
    #   Estimate return regression WLS-EV beta using Johnson
    #
    # METHODS:
    #   'init'               : Initialization of the object with the es50 log return and estimated volatility data
    #   'estimate_wlf_ev'  : Estimate return regression beta WLS-EV using Johnson (2016)
    #

    def __init__(self, log_returns):
        #
        # DESCRIPTION:
        #   Initialize object with es50 data
        #
        self.log_returns = log_returns['logreturns']
        self.est_var = log_returns['vol_daily_est']

        print('WLS-EV Regression Object initialized!')

    def estimate_wlf_ev(self):
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
        # TODO: create shifted series

        # delete last row of series to adjust dimensionality
        X_log_rets_dim_adj = self.log_returns[:-1]
        X_est_var_dim_adj  = self.est_var[:-1]

        # X = X_t/sigma2, no constant since constant is already in X_t
        X =X_log_rets_dim_adj/X_est_var_dim_adj

        print("X head: {}".format(X.head()))


        # Shift Y data up to get (t+1) and delete last row of series to delete nan
        Y_log_returns_shift = self.log_returns.shift(-1) # shift up
        Y_log_returns_shift = Y_log_returns_shift[:-1] # delete last row

        Y_est_var_shift = self.est_var.shift(-1) # shift up
        Y_est_var_shift = Y_est_var_shift[:-1] # delete last row

        # Y = r_(t+1)/sigma2
        Y = Y_log_returns_shift/Y_est_var_shift

        print("Y head: {}".format(Y.head()))

        # Next, run ols regression to estimate the wlsev parameters
        #
        wlsev_reg_model = sm.OLS(Y, X)
        wlsev = wlsev_reg_model.fit()  # Fit the model

        # TODO: Distinguish between overlapping returns and exception one day ahead prediction (non-overlapping returns)

        # Error Statistics
        # --------------------------------------------------------------------------------
        # Get robust standard errors Newey West (1987) with 6 lags
        robust_standard_errors = wlsev.get_robustcov_results(cov_type='HAC', maxlags=6)
        robust_standard_errors.summary()

        return wlsev, robust_standard_errors
