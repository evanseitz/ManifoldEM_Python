import numpy as np
import R_p
import a
'''
function tau=solve_d_R_d_tau_p_3D()
% 
% tau = solve_d_R_d_tau_p_3D()
% 
% returns the value of tau for data point p that minimizes the residual
% R to the model x_ij = a_j cos(j*pi*tau_i) + b_j for i = p.
% 
% j goes from 1 to 3 (this is only for 3D systems).
% 
% i goes from 1 to nS where nS is the number of data points to be fitted.
% 
% For a fixed set of a_j and b_j, j=1:3, tau_i for i=1:nS are
% obtained by putting dR/d(tau_i) to zero.
% 
% For a fixed set of tau_i, i=1:nS, a_j and b_j for j=1:3 are
% obtained by solving 3 sets of 2x2 linear equations.
% 
% Global input:
%   p: index of the data point of interest.
%   a, b: dimensions 1 x 3, the current set of model coefficients.
%   x: dimensions nS x 3, the data points to be fitted.
% Output:
%   tau: tau value for data point p that best fits the model.
% 
% copyright (c) Russell Fung 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    

'''

def op():   #added
    #global p, a, b, x
    d_R_d_beta_3D = np.array([48*a.a[2]**2,
                              0,
                              8*a.a[1]**2-48*a.a[2]**2,
                              -12*a.a[2]*(a.x[a.p,2]-a.b[2]),
                               a.a[0]**2-4*a.a[1]**2+9*a.a[2]**2-4*a.a[1]*(a.x[a.p,1]-a.b[1]),
                               -a.a[0]*(a.x[a.p,0]-a.b[0])+3*a.a[2]*(a.x[a.p,2]-a.b[2])]).T
    beta = np.roots(d_R_d_beta_3D)
    #print 'd_R is', d_R_d_beta_3D
    # remove elements for which tmp is true
    tmp = np.absolute(np.imag(beta))>0
    tmp = np.nonzero(tmp)[0]
    #print 'beta is', beta
    beta1 = np.delete(beta, tmp, None)
    # remove elements for which tmp is true
    tmp = np.absolute(beta1)>1
    tmp = np.nonzero(tmp)[0]
    beta = np.real(np.delete(beta1, tmp, None))
    #print 'beta after is', beta
    #
    #print 'beta shape is',beta.shape
    tau_candidate = np.vstack((np.arccos(beta.reshape(-1,1))/np.pi,0,1))
    id = np.argmin(R_p.op(tau_candidate))
    tau = tau_candidate[id]
    return tau,beta
