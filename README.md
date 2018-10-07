# Least squares weighted by ex-ante returns 

This is a project about robust linear forecasting with crash risk measures, following Johnson (2018). Code and seminar thesis are the result of a seminar project with the Computational Risk and Asset Management Group at the KIT .

OLS is easy to use and the most efficient linear unbiased estimator when error terms are homoscedastic and without autocorrelation (Johnson 2018). However, these assumptions rarely hold in practice when working with financial data, resulting in inefficient model estimations with more error than necessary.

Johnson (2018) provides a GLS method which uses ex-ante return variance as weights. This is, in the case of return predictability regressions, easy to use and results in large efficiency gains. It is called WLS-EV. Efficiency gains come from scaling regression residuals by an ex ante variance estimate, which makes different volatile observations comparable in terms of signal versus noise.

The goal of this project is to implement a python library which provides parameter and standard error estimates of WLS-EV. The library will then be applied to financial data. Our results are compared to the conclusions of Johnson (2018) and further applications, especially the transfer from return to non-return linear relationships, are discussed.
