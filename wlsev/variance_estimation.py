__author__ = 'Tobias Kuhlmann'

import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.formula.api as smf

import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')


class ExAnteVariance(object):

    """
    This class implements the first wls-ev step:
      Estimate  sigma_t^2, the conditional variance of next-period unexpected returns ✏t+1.

    METHODS:
      'init'               : Initialization of the object with the es50 volatility data
      'estimate_variance'  : Estimate (sigma_t)2, the (ex ante) conditional variance of next-period unexpected
                             returns epsilon_(t+1) using a HAR-RV (Hierachical Autoregressive-Realized Variance)
                             Model from Corsi (2009)
    """

    def __init__(self, vol, implied_vol=None):
        #
        # DESCRIPTION:
        #   Initialize object with es50 volatility
        #
        self.vol = vol
        self.imp_vol = implied_vol
        self.ols_res = None
        self.plt = None

        print('Variance Estimation Object initialized!')

    def estimate_variance(self):
        # 1. Estimate (sigma_t)2, the (ex ante) conditional variance of next-period unexpected returns epsilon_(t+1)
        # using a HAR-RV (Hierachical Autoregressive-Realized Variance) Model from Corsi (2009)
        print('Variance Estimation begins!')
        # dependent variable: shift vol_daily as dependent variable
        self.vol = pd.DataFrame({'var_daily': self.vol})
        self.vol['var_daily_est'] = self.vol.shift(-1)
        # explanatory variable 2: calculate rolling average volatility weekly
        self.vol['var_weekly'] = self.vol['var_daily'].rolling(window=5, center=False).mean()
        # explanatory variable3: calculate rolling average volatility monthly
        self.vol['var_monthly'] = self.vol['var_daily'].rolling(window=22, center=False).mean()

        # delete rows with nan, resulting from weekly and monthly rolling mean at the end
        self.vol = self.vol.dropna()

        # Check if implied vol is passed and proceed respectively
        # Without implied vol
        if self.imp_vol is None:
            # Corsi (2009) HAR-EV Model: RV_(t+1d)(d)=c+beta(d)*RV_t(d)+beta(w)*RV_t(w)+beta(m)*RV_t(m)+w_(t+1d)(d)
            # fit regression model
            self.ols_res = smf.ols(formula="var_daily_est ~ var_daily + var_weekly + var_monthly", data=self.vol).fit()
            print("Variance Estimation Results")
            print(self.ols_res.summary())
            # predict with fitted regression model
            self.vol['var_daily_est'] = self.ols_res.predict(self.vol[['var_daily', 'var_weekly', 'var_monthly']])
            print('Variance estimated!')
        # With implied vol
        else:
            # Corsi (2009) HAR-EV Model: RV_(t+1d)(d)=c+beta(d)*RV_t(d)+beta(imp)*IMPV_t(d)+beta(w)*RV_t(w)+beta(m)*RV_t(m)+w_(t+1d)(d)
            # fit regression model
            self.vol = self.vol.join(self.imp_vol)
            self.ols_res = smf.ols(formula="var_daily_est ~ var_daily + implied_var + var_weekly + var_monthly",
                                   data=self.vol).fit()
            print("Variance Estimation Results")
            print(self.ols_res.summary())
            # predict with fitted regression model
            self.vol['var_daily_est'] = self.ols_res.predict(self.vol[['var_daily', 'implied_var', 'var_weekly', 'var_monthly']])
            print('Variance estimated!')
        return self.vol['var_daily_est']
