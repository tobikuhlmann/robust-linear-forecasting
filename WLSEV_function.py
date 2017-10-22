
#
# ==================================================
# CRAM Seminar WS17/18
# @author Tobias Kuhlmann
# Algorithmic Design - Robust Estimators with Weighted Least Squares Estimates
# ==================================================
#

# Import packages
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

# Data Import
# --------------------------------------------------

# Import price data
es_50_prices = pd.read_csv('es50_prices.csv', parse_dates = True)
es_50_prices.head

# Delete unnecessary columns
del es_50_prices['openprice']
del es_50_prices['highprice']
del es_50_prices['lowprice']
del es_50_prices['volume']

# set index, rename and check

es_50_prices = es_50_prices.set_index('loctimestamp')
es_50_prices = es_50_prices.rename(columns = {'loctimestamp': 'date'} )
es_50_prices.tail()


