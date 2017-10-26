
# Imports

# Import packages for econometric analysis
import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.formula.api as smf

# Import plotting library
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


class ExAnteVariance(object):
    #
    # DESCRIPTION:
    #   This class implements the first wls-ev step:
    #   Estimate  sigma_t^2, the conditional variance of next-period unexpected returns ✏t+1.
    #
    # METHODS:
    #   'init'               : Initialization of the object with the es50 volatility data
    #   'estimate_variance'  : Estimate (sigma_t)2, the (ex ante) conditional variance of next-period unexpected
    #                          returns epsilon_(t+1) using a HAR-RV (Hierachical Autoregressive-Realized Variance)
    #                          Model from Corsi (2009)
    #

    def __init__(self, vol):
        #
        # DESCRIPTION:
        #   Initialize object with es50 volatility
        #
        self.vol = vol
        self.nan = None
        self.ols_res = None
        self.plt = None

        print('Variance Estimation Object initialized!')

    def estimate_variance(self):
        # 1. Estimate (sigma_t)2, the (ex ante) conditional variance of next-period unexpected returns epsilon_(t+1)
        # using a HAR-RV (Hierachical Autoregressive-Realized Variance) Model from Corsi (2009)

        print('Variance Estimation begins!')
        # explanatory variable 1: rename daily volatility
        self.vol = self.vol.rename(columns={'volatility': 'vol_daily'})
        # dependent variable: shift vol_daily as dependent variable
        self.vol['vol_daily_est'] = self.vol['vol_daily'].shift(-1)
        # explanatory variable 2: calculate rolling average volatility weekly
        self.vol['vol_weekly'] = self.vol['vol_daily'].rolling(window=5, center=False).mean()
        # explanatory variable3: calculate rolling average volatility monthly
        self.vol['vol_monthly'] = self.vol['vol_daily'].rolling(window=22, center=False).mean()
        print(self.vol.head())

        # Detect, delete print rows with nan
        self.nan = self.vol[self.vol.isnull().any(axis=1)]
        self.vol = self.vol.dropna()
        print("{} rows with nan detected and deleted".format(self.nan.shape[0]))
        print("New dataframe shape {}".format(self.vol.shape))

        # Corsi (2009) HAR-EV Model: RV_(t+1d)(d)=c+beta(d)*RV_t(d)+beta(w)*RV_t(w)+beta(m)*RV_t(m)+w_(t+1d)(d)
        # fit regression model
        self.ols_res = smf.ols(formula="vol_daily_est ~ vol_daily + vol_weekly + vol_monthly", data=self.vol).fit()
        print(self.ols_res.summary())
        # predict with fitted regression model
        self.vol['vol_daily_est'] = self.ols_res.predict()
        print('Variance estimated!')

        # plot
        self.vol.plot(subplots=True)

        return self.vol['vol_daily_est']
