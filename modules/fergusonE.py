import numpy as np, logging
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
warnings.simplefilter(action='ignore', category=OptimizeWarning)
import time

"""
%--------------------------------------------------------------------------
% function ferguson(D,s)
% D: Distance matrix
% logEps: Range of values to try
% Adapted from Chuck, 2011
%--------------------------------------------------------------------------
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)    
%
"""
def fun(xx, aa0, aa1, aa2, aa3):
    F = aa3 + aa2 * np.tanh(aa0 * xx + aa1)
    return F

def find_thres(logEps, D2):
    eps = np.exp(logEps)
    d = 1. / (2. * np.max(eps)) * D2
    sg = np.sort(d)
    ss = np.sum(np.exp(-sg))
    thr = max(-np.log(0.01 * ss / len(D2)), 10)  # taking 1% of the average (10)
    return thr

def op(D,logEps,a0):
    # range of values to try:
    logSumWij = np.zeros(len(logEps))
    D2 = D * D
    thr = find_thres(logEps, D2)
    for k in range(len(logEps)):
        eps = np.exp(logEps[k])
        d = 1. / (2. * eps) * D2
        d = -d[d < thr]
        Wij = np.exp(d) #see Coifman 2008
        logSumWij[k] = np.log(sum(Wij))

    # curve fitting of a tanh():
    resnorm = np.inf
    cc = 0
    while (resnorm>100):
        cc += 1
        popt, pcov = curve_fit(fun, logEps, logSumWij, p0=a0)
        resnorm = sum(np.sqrt(np.fabs(np.diag(pcov))))
        a0 = 1 * (np.random.rand(4, 1) - .5)
        
        residuals = logSumWij - fun(logEps, popt[0], popt[1], popt[2], popt[3])
        ss_res = np.sum(residuals**2) #residual sum of squares
        ss_tot = np.sum((logSumWij - np.mean(logSumWij))**2) #total sum of squares
        R_squared = 1 - (ss_res / ss_tot) #R**2-value

    return (popt, logSumWij, resnorm, R_squared)
