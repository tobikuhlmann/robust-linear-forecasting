__author__ = 'Tobias Kuhlmann'

from variance_estimation import ExAnteVariance
from wlsev_model import Wlsev_model
from ols_model import OLS_model
import visualisation
from simon_ols_model import OLS
import numpy as np
import pandas as pd
import matplotlib

# Data Import and Transformation
# ==================================================

# Import price data and calc log returns
# --------------------------------------------------
'''Simulated data a = 1; b = 1; sigma = 0.5 at forecast horizon 5'''
retvol = pd.read_csv('data/simulated.csv', sep=";")
# Calculate variance from vol
retvol['volatility'] = retvol['volatility'] ** 2



# Model and Analysis
# ==================================================

# 2. least squares estimates weighted by ex-ante return variance (WLS-EV) using Johnson (2016)
# ------------------------------------------------------------------------------------------------------------

# set forecast_horizon
forecast_horizon = 5
# Instantiate object
wlsev_obj = Wlsev_model(retvol['r'][:-1].as_matrix(), retvol['r'][1:].as_matrix(), retvol['volatility'][1:].as_matrix(), forecast_horizon)

# fit model
wlsev_obj.fit()
wlsev_obj.evaluate()
wlsev_obj.print_results()
wlsev_obj.plot_results()
wlsev_obj.plot_scatter()
# get data
X, Y, y_wlsev = wlsev_obj.get_plot_data_wlsev()

# Instantiate object
ols_obj = OLS_model(retvol['r'][:-1].as_matrix(), retvol['r'][1:].as_matrix(), forecast_horizon)

# fit model
ols_obj.fit()
ols_obj.evaluate()
ols_obj.print_results()
ols_obj.plot_results()
ols_obj.plot_scatter()
# get data
X, Y, y_ols = ols_obj.get_plot_data_ols()


# Visualisation
# ------------------------------------------------------------------------------------------------------------

# time series plot
visualisation.plot_results(X,Y,y_wlsev, y_ols)
# scatter plot
visualisation.plot_scatter(X,Y,y_wlsev, y_ols)