
# Imports

# Import packages for econometric analysis
import numpy as np
import pandas as pd
import statsmodels as sm

# Import plotting library
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')



class WlsEvEstimation(object):

    # DESCRIPTION:
    #   This class implements the second wls-ev step:
    #   Estimate return regression WLS-EV beta using Johnson
    #
    # METHODS:
    #   'init'               : Initialization of the object with the es50 log return and estimated volatility data
    #   'estimate_wlf_ev'  : Estimate return regression beta WLS-EV using Johnson (2016)
    #

    def __init__(self, log_returns):
        #
        # DESCRIPTION:
        #   Initialize object with es50 data
        #
        self.log_returns = log_returns['logreturns']
        self.est_vol = log_returns['volatility']

        print('WLS-EV Regression Object initialized!')

    def estimate_wlf_ev(self):
        #
        # DESCRIPTION:
        #   Estimate return regression beta with WLS-EV
        #

        # import libraries
        import pandas as pd
        import statsmodels as sm

        print(self.log_returns.head())
        print(self.est_vol.head())


# Error Statistics
# --------------------------------------------------
