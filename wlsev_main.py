
#
# ============================================================================
# CRAM Seminar WS17/18
# @author Tobias Kuhlmann
# Algorithmic Design - Least squares estimates weighted by ex-ante return variance (WLS-EV)
# ============================================================================
#

# Import packages
from variance_estimation import ExAnteVariance
from wlsev_estimation import Wlsevestimation
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')



# Data Import and Transformation
# --------------------------------------------------
# Import price data and calc log returns
# --------------------------------------------------
es_50_prices = pd.read_csv('es50_prices.csv', parse_dates = True)
es_50_prices.head()

# Delete unnecessary columns
del es_50_prices['openprice']
del es_50_prices['highprice']
del es_50_prices['lowprice']
del es_50_prices['volume']
del es_50_prices['instrumentid']

# set index, rename and check
es_50_prices = es_50_prices.rename(columns={'loctimestamp': 'date'})
es_50_prices = es_50_prices.set_index('date')
es_50_prices.sort_index()
es_50_prices.head()

# Log Returns
es_50_logret = es_50_prices
es_50_logret['logreturns'] = np.log(es_50_prices['lastprice'] / es_50_prices['lastprice'].shift(1))
es_50_logret.head()


# Import vol data
# --------------------------------------------------
es_50_vol = pd.read_csv('es50_volatility.csv', parse_dates=True)
# Transform dates
es_50_vol['loctimestamp'] = pd.to_datetime(es_50_vol['loctimestamp'])

# Delete unnecessary columns
del es_50_vol['instrumentid']

# Calculate variance from vol
es_50_vol['volatility'] = es_50_vol['volatility'] ** 2

# set index, rename and check
es_50_vol = es_50_vol.rename(columns={'loctimestamp': 'date'})
es_50_vol = es_50_vol.set_index('date')
es_50_vol = es_50_vol.sort_index()


# Import implied volatility
# --------------------------------------------------
es_50_imp_vol = pd.read_csv('es50_implied_volatility.csv', parse_dates=True)

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
es_50_imp_vol = es_50_imp_vol.sort_index()

# join vol and implied vol
es_50_imp_vol = es_50_vol.join(es_50_imp_vol['implied_vol']).dropna()

# Main function
# ---------------------------------------------------------------------------------------------------------------------
#

# 1. Estimate (sigma_t)2, the (ex ante) conditional variance of next-period unexpected returns epsilon_(t+1)
# using a HAR-RV (Hierachical Autoregressive-Realized Variance) Model from Corsi (2009)
# ------------------------------------------------------------------------------------------------------------
# First, instantiate object
# no implied vol
ea_var_obj = ExAnteVariance(es_50_vol)
# implied vol exists
#ea_var_obj = ExAnteVariance(es_50_imp_vol, es_50_imp_vol['implied_vol'])

# Estimate Variance
result = ea_var_obj.estimate_variance()
result = result.dropna()
print('result')
print("Estimated variance: {}".format(result.head()))


# 2. least squares estimates weighted by ex-ante return variance (WLS-EV) using Johnson (2016)
# ------------------------------------------------------------------------------------------------------------
# First, instantiate object
wlsev_var_rets = es_50_logret.join(result).dropna()
wlsev_var_rets.plot(subplots=True)
# set forecast_horizon
forecast_horizon = 10
wlsev_obj = Wlsevestimation(wlsev_var_rets, forecast_horizon)

# for non-overlapping day-ahead prediction
#wlsev, robust_standard_errors = wlsev_obj.estimate_wlf_ev_non_overlapping()
#wlsev, robust_standard_errors = wlsev_obj.estimate_wlf_ev_non_overlapping()


# for overlapping interval-ahead prediction with forecast horizon h
wlsev, robust_standard_errors = wlsev_obj.estimate_wlf_ev_overlapping()

print(robust_standard_errors.summary())





