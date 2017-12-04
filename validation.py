
# Import packages
from variance_estimation import ExAnteVariance
from wlsev_model import Wlsev_model
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
retvol = pd.read_csv('simulated.csv')
retvol

# Calculate variance from vol
retvol['volatility'] = retvol['volatility'] ** 2

# Model and Analysis
# ==================================================

# 2. least squares estimates weighted by ex-ante return variance (WLS-EV) using Johnson (2016)
# ------------------------------------------------------------------------------------------------------------
# Join returns and estimated variance
# set forecast_horizon
forecast_horizon = 1
# Instantiate object
wlsev_obj = Wlsev_model(retvol['r'].as_matrix(), retvol['volatility'].as_matrix(), forecast_horizon)

# fit model
betas, std_errors, t_stats = wlsev_obj.estimate_wls_ev()

# OOS evaluation to get MSEs and Rsquared
#r_squared = wlsev_obj.wls_ev_eval()
