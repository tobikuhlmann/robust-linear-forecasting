__author__ = 'Tobias Kuhlmann'

from variance_estimation import ExAnteVariance
from wlsev_model import Wlsev_model
from ols_model import OLS_model
from simon_ols_model import OLS
import numpy as np
import pandas as pd
import matplotlib

# Data Import and Transformation
# ==================================================

# Import price data and calc log returns
# --------------------------------------------------
retvol = pd.read_csv('data/simulated.csv', sep=";")
retvol

# Calculate variance from vol
retvol['volatility'] = retvol['volatility'] ** 2

# Model and Analysis
# ==================================================

# 2. least squares estimates weighted by ex-ante return variance (WLS-EV) using Johnson (2016)
# ------------------------------------------------------------------------------------------------------------
# Join returns and estimated variance
# set forecast_horizon
forecast_horizon = 5
# Instantiate object
wlsev_obj = Wlsev_model(retvol['r'][:-1].as_matrix(), retvol['r'][1:].as_matrix(), retvol['volatility'][:-1].as_matrix(), forecast_horizon)
# fit model
wlsev_obj.fit()
wlsev_obj.evaluate()
wlsev_obj.print_results()
wlsev_obj.plot_results()

# Instantiate object
ols_obj = OLS_model(retvol['r'][:-1].as_matrix(), retvol['r'][1:].as_matrix(), forecast_horizon)
# fit model
ols_obj.fit()
ols_obj.evaluate()