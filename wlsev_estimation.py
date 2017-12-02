
# Imports

# Import packages for econometric analysis
import numpy as np
import pandas as pd
import statsmodels as sm

# Import plotting library
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')



class Wlsevestimation(object):

    # DESCRIPTION:
    #   This class implements the second wls-ev step:
    #   Estimate return regression WLS-EV beta using Johnson
    #
    # METHODS:
    #   'init'               : Initialization of the object with the es50 log return and estimated volatility data
    #   'estimate_wlf_ev_non_overlapping'  : Estimate return regression beta WLS-EV using Johnson (2016) for non overlapping returns
    #   'estimate_wlf_ev_overlapping'  : Estimate return regression beta WLS_EV using Johnson (2016) and Hodrick (1992) to account for overlapping returns
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

    def estimate_wlf_ev_non_overlapping(self):
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
        est_var_dim_adj  = self.est_var[:-1].as_matrix()**0.5

        # X = X_t/sigma2_t, no constant since constant is already in X_t, delete last row to adjust for dimensionality
        X = self.log_returns[:-1].as_matrix()/est_var_dim_adj

        # Y = r_(t+1)/sigma2_t
        Y = self.log_returns[1:].as_matrix()/est_var_dim_adj

        # Next, run ols regression to estimate the wlsev parameters
        wlsev_reg_model = sm.OLS(Y, X)
        wlsev = wlsev_reg_model.fit()  # Fit the model


        # Error Statistics
        # --------------------------------------------------------------------------------
        # Get robust standard errors Newey West (1987) with 6 lags
        robust_standard_errors = wlsev.get_robustcov_results(cov_type='HAC', maxlags=6)
        robust_standard_errors.summary()

        return wlsev, robust_standard_errors

    def estimate_wlf_ev_overlapping(self):
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
        est_var_dim_adj  = self.est_var[self.forecast_horizon-1:-1].as_matrix()**0.5

        # X = HodrickSum(X)/sigma2_t, no constant since constant is already in X
        # Initialize X(t) and divide by estimated sigma_t->t+1 for wls_ev
        #X = self.log_returns[self.forecast_horizon-1:-1].as_matrix() / est_var_dim_adj
        X = self.log_returns[self.forecast_horizon - 1:-1].as_matrix()

        # Stack X and X(t-1) + X(t-2) + ... + X(t-(forecast horizon-1)) and divide each instance by estimated sigma_t->t+1 for wls_ev
        for i in range(1, self.forecast_horizon):
            #X = np.vstack((X, self.log_returns[(self.forecast_horizon-(1+i)):-(1+i)].as_matrix() / est_var_dim_adj))
            X = np.vstack((X, self.log_returns[(self.forecast_horizon - (1 + i)):-(1 + i)].as_matrix()))

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
        Y = self.log_returns[self.forecast_horizon:].as_matrix() / est_var_dim_adj
        Y = np.transpose(Y)

        # Next, run ols regression to estimate the wlsev parameters
        wlsev_reg_model = sm.OLS(Y, X)
        wlsev = wlsev_reg_model.fit()  # Fit the model


        # Error Statistics
        # --------------------------------------------------------------------------------
        # Get robust standard errors Newey West (1987) with 6 lags
        robust_standard_errors = wlsev.get_robustcov_results(cov_type='HAC', maxlags=6)
        #robust_standard_errors.summary()

        # TODO: 2. Scale the resulting coefficients and standard errors

        # Scale Var_x_t_rolling/Var_x_t
        scale = Var_x_t_rolling / Var_x_t

        # Scale betas
        betas = scale * robust_standard_errors.params

        # Scale standard errors
        std_errors = scale * robust_standard_errors.bse

        # t-statistic
        t_stats = betas / std_errors

        # Print Results
        print("WLS-EV Estimation Results")
        print("-------------------------------------------------------------------------------------------------------")
        print("Scale: {}".format(scale))
        print("Scaled betas: {}".format(betas))
        print("Scaled bse standard errors: {}".format(std_errors))
        print("Scaled t-stats: {}".format(t_stats))
        print("-------------------------------------------------------------------------------------------------------")

        return betas, std_errors, t_stats



    def predict_wls_ev(self):
        #
        # DESCRIPTION:
        #   Estimate return regression beta with WLS-EV
        #

        # import libraries
        import pandas as pd
        import statsmodels.api as sm



        return wlsev, robust_standard_errors
