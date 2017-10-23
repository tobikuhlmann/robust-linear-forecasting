
#
# ============================================================================
# CRAM Seminar WS17/18
# @author Tobias Kuhlmann
# Algorithmic Design - Least squares estimates weighted by ex-ante return variance (WLS-EV)
# ============================================================================
#

# Import packages
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

# Data Import and Transformation
# --------------------------------------------------

# Import price data
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

# Import vol data
es_50_vol = pd.read_csv('es50_volatility.csv', parse_dates=True)
# Transform dates
es_50_vol['loctimestamp'] = pd.to_datetime(es_50_vol['loctimestamp'])

# Delete unnecessary columns
del es_50_vol['instrumentid']

# set index, rename and check
es_50_vol = es_50_vol.rename(columns={'loctimestamp': 'date'})
es_50_vol = es_50_vol.set_index('date')
es_50_vol.sort_index()

# Join prices and vol
es_50 = es_50_prices.join(es_50_vol)
es_50.head()
# shape test
if es_50.shape[0]==es_50_prices.shape[0] and es_50.shape[0]==es_50_vol.shape[0]:
    print('Data Import and Join successfull')

# Log Returns
# --------------------------------------------------
es_50['logreturns'] = np.log(es_50['lastprice'] / es_50['lastprice'].shift(1))
es_50.head()

# Overview Plot
es_50.plot(subplots = True)

# Main function
# least squares estimates weighted by ex-ante return variance (WLS-EV) using Johnson (2016)
# --------------------------------------------------


# 1. Estimate (sigma_t)2, the (ex ante) conditional variance of next-period unexpected returns epsilon_(t+1)
# using a HAR-RV (Hierachical Autoregressive-Realized Variance) Model from Corsi (2009)
# ------------------------------------------------------------------------------------------------------------
def estimate_variance(vol):
    # import libraries
    import pandas as pd
    import statsmodels.formula.api as smf

    # explanatory variable 1: rename daily volatility
    vol = vol.rename(columns={'volatility': 'vol_daily'})
    # dependent variable: shift vol_daily as dependent variable
    vol['vol_daily_est'] = vol['vol_daily'].shift(-1)
    # explanatory variable 2: calculate rolling average volatility weekly
    vol['vol_weekly'] = vol['vol_daily'].rolling(window=5,center=False).mean()
    # explanatory variable3: calculate rolling average volatility monthly
    vol['vol_monthly'] = vol['vol_daily'].rolling(window=22, center=False).mean()
    # Detect, delete print rows with nan
    nan=vol[vol.isnull().any(axis=1)]
    vol = vol.dropna()
    print("{} rows with nan detected and deleted".format(nan.shape[0]))
    print("New dataframe shape {}".format(vol.shape))
    # Corsi (2009) HAR-EV Model: RV_(t+1d)(d)=c+beta(d)*RV_t(d)+beta(w)*RV_t(w)+beta(m)*RV_t(m)+w_(t+1d)(d)
    # fit regression model
    olsres = smf.ols(formula="vol_daily_est ~ vol_daily + vol_weekly + vol_monthly", data=vol).fit()
    print(olsres.summary())
    # predict with fitted regression model
    vol['vol_daily_est'] = olsres.predict()
    #print(vol.head())
    return(vol['vol_daily_est'])


# 2. Estimate regression WLS-EV beta using Johnson (2016)
# ------------------------------------------------------------------------------------------------------------
def estimate_wlf_ev(vol_est, returns):
    # import libraries
    import pandas as pd
    import statsmodels as sm

    variance.head()
    returns.head()

# Error Statistics
# --------------------------------------------------


# =======Run=======
vol_est = estimate_variance(es_50_vol)


