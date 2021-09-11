import numpy as np
import a

'''
Copyright (c) Columbia University Hstau Liao 2019    
'''

'''
function fval=R_p(tau_p)
% 
% fval = R_p(tau_p)
% 
% returns the fitting residual R to the model
% x_ij = a_j cos(j*pi*tau_i) + b_j for i = p, and the given value of tau_p.
% 
% j goes from 1 to nDim where nDim is the dimension of the system.
% 
% i goes from 1 to nS where nS is the number of data points to be fitted.
% 
% For a fixed set of a_j and b_j, j=1:nDim, tau_i for i=1:nS are
% obtained by putting dR/d(tau_i) to zero.
% 
% For a fixed set of tau_i, i=1:nS, a_j and b_j for j=1:nDim are
% obtained by solving nDim sets of 2x2 linear equations.
% 
% Global input:
%   p: index of the data point of interest.
%   nDim: dimension of the system.
%   a, b: dimensions 1 x nDim, the current set of model coefficients.
%   x: dimensions nS x nDim, the data points to be fitted.
% Input:
%   tau_p: current choice of tau for data point index p.
% Output:
%   fval: R_p calculated at tau_p.
% 
% copyright (c) Russell Fung 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
'''

def op(tau_p):
    #global p, nDim, a, b, x
    jj = np.array(range(1,a.nDim+1))
    j_pi_tau_p = tau_p*jj*np.pi
    a_cos_j_pi_tau_p = a.a*np.cos(j_pi_tau_p)
    err = a.x[a.p,:]-a.b - a_cos_j_pi_tau_p
    fval = np.sum(err**2,axis=1)

    return fval
