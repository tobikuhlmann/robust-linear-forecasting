
#
# ============================================================================
# CRAM Seminar WS17/18
# @author Tobias Kuhlmann
# Algorithmic Design - Robust Estimators with Weighted Least Squares Estimates
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


# Estimating beta using WLS-EV
# --------------------------------------------------
# 1. Estimate  (sigma_t)2, the (ex ante) conditional variance of next-period unexpected returns epsilon_(t+1)
# 1.1 Calculate Daily, Weekly (average) and Monthly (average) aggregated volatilities

# 2. Estimate regression beta_WLS-EV using:


# Error Statistics
# --------------------------------------------------




