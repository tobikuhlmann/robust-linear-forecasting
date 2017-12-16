"""
KIT CRAM Seminar WS17/18
Algorithmic Design - Least squares estimates weighted by ex-ante return variance (WLS-EV)
"""

__author__ = 'Tobias Kuhlmann'

# Import packages
from variance_estimation import ExAnteVariance
from wlsev_model import Wlsev_model
from ols_model import OLS_model
from simon_ols_model import OLS


import numpy as np
import pandas as pd
import matplotlib

matplotlib.use
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

# Data Import and Transformation
# ==================================================

# Import price data and calc log returns
# --------------------------------------------------
es_50_prices = pd.read_csv('data/es50_prices.csv', parse_dates=True)

# Delete unnecessary columns
del es_50_prices['openprice']
del es_50_prices['highprice']
del es_50_prices['lowprice']
del es_50_prices['volume']
del es_50_prices['instrumentid']

# set index, rename and check
es_50_prices = es_50_prices.rename(columns={'loctimestamp': 'date'})
es_50_prices = es_50_prices.set_index('date')

# Log Returns
es_50_logret = es_50_prices
es_50_logret['logreturns'] = np.log(es_50_prices['lastprice'] / es_50_prices['lastprice'].shift(1))

# Import vol data
# --------------------------------------------------
es_50_vol = pd.read_csv('data/es50_volatility.csv', parse_dates=True)
# Transform dates
es_50_vol['loctimestamp'] = pd.to_datetime(es_50_vol['loctimestamp'])

# Delete unnecessary columns
del es_50_vol['instrumentid']

# Calculate variance from vol
es_50_vol['volatility'] = es_50_vol['volatility'] ** 2

# set index, rename and check
es_50_vol = es_50_vol.rename(columns={'loctimestamp': 'date'})
es_50_vol = es_50_vol.set_index('date')

# Import implied volatility
# --------------------------------------------------
es_50_imp_vol = pd.read_csv('data/es50_implied_volatility.csv', parse_dates=True)

# Transform dates
es_50_imp_vol['loctimestamp'] = pd.to_datetime(es_50_imp_vol['loctimestamp'])

# Delete unnecessary columns
del es_50_imp_vol['instrumentid']
del es_50_imp_vol['maturity']

# Calculate implied variance from implied vol
es_50_imp_vol['implied_vol'] = es_50_imp_vol['measure'] ** 2

# set index, rename and check
es_50_imp_vol = es_50_imp_vol.rename(columns={'loctimestamp': 'date'})
es_50_imp_vol = es_50_imp_vol.set_index('date')

# join vol and implied vol
es_50_imp_vol = es_50_vol.join(es_50_imp_vol['implied_vol']).dropna()

# Model and Analysis
# ==================================================
#
# 1. Estimate (sigma_t)2, the (ex ante) conditional variance of next-period unexpected returns epsilon_(t+1)
# using a HAR-RV (Hierachical Autoregressive-Realized Variance) Model from Corsi (2009)
# ------------------------------------------------------------------------------------------------------------
# First, instantiate object
# no implied vol
ea_var_obj = ExAnteVariance(es_50_vol)
# implied vol exists
# ea_var_obj = ExAnteVariance(es_50_imp_vol, es_50_imp_vol['implied_vol'])

# Estimate Variance
result = ea_var_obj.estimate_variance()
result = result.dropna()
# Join returns and estimated variance
wlsev_var_rets = es_50_logret.join(result).dropna()


# 2. least squares estimates weighted by ex-ante return variance (WLS-EV) using Johnson (2016)
# ------------------------------------------------------------------------------------------------------------

# set forecast_horizon
forecast_horizon = 1

# WLS-EV
wlsev_obj = Wlsev_model(wlsev_var_rets['logreturns'][:-1].as_matrix(), wlsev_var_rets['logreturns'][1:].as_matrix(), wlsev_var_rets['vol_daily_est'][:-1].as_matrix(), forecast_horizon)
wlsev_obj.fit()
# OOS evaluation to get Rsquared
wlsev_obj.evaluate()
wlsev_obj.print_results()

# OLS
ols_obj = OLS_model(wlsev_var_rets['logreturns'][:-1].as_matrix(), wlsev_var_rets['logreturns'][1:].as_matrix(), forecast_horizon)
ols_obj.fit()
# OOS evaluation to get Rsquared
ols_obj.evaluate()
ols_obj.print_results()

# Get Simon's OLS estimation results
if forecast_horizon == 1:
    ols_model = OLS(wlsev_var_rets['logreturns'][:-1].as_matrix(), wlsev_var_rets['logreturns'][1:].as_matrix())
    ols_model.fit()
    ols_model.printResults()

