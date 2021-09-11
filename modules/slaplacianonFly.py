import numpy as np
import time
from scipy.sparse import csc_matrix
from collections import namedtuple

"""
function [ l, sigmaTune ] = slaplacianonFly( yVal,yCol,yRow, nS, varargin )
%SLAPLACIAN  Sparse Laplacian matrix
%
% Given a set of nS data points, and the dinstances to nN nearest neighbors
% for each data point, slaplacian computes a sparse, nY by nY symmetric
% graph Laplacian matrix l.
%
% The input data are supplied in the column vectors yVal and yInd of length
% nY * nN such that
%
%   yVal( ( i - 1 ) * nN + ( 1 : nN ) ) contains the distances to the
%   nN nearest neighbors of data point i sorted in ascending order, and
%
%   yInd( ( i - 1 ) * nN + ( 1 : nN ) ) contains the indices of the nearest
%   neighbors.
%
%   yVal and yInd can be computed by calling nndist
%
%   slaplacian admits a number of options passed as name-value pairs
%
%   alpha : normalization, according to Coifman & Lafon
%
%   nAutotune : number of nearest neighbors for autotuning. Set to zero if no
%   autotuning is to be performed
%
%   sigma: width of the Gaussian kernel
%
%
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2019 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)   

"""

def op(*arg):
    yVal = arg[0]
    yCol = arg[1]
    yRow = arg[2]
    nS = arg[3] #dataset size
    options = arg[4] #options.sigma: Gaussian width 
    #print('sigma:', options.sigma)
    nNZ = len(yVal) #number of nonzero elements

    # if required, compute autotuning distances:
    if options.autotune > 0:
        print('Autotuning is not implemented in this version of slaplacian' + '\n')
    else:
        sigmaTune = options.sigma

    yVal = yVal / sigmaTune**2

    # compute the unnormalized weight matrix:
    yVal = np.exp( -yVal ) #apply exponential weights (yVal is distance**2)
    l = csc_matrix((yVal,(yRow, yCol)),shape = (nS, nS))

    d = sum(l)
    d = d.toarray()
    if options.alpha != 1: #apply non-isotropic normalization
        d = d**options.alpha
    d = d.T
    yVal = yVal / (d[yRow].flatten('C') * d[yCol].flatten('C'))
    l = csc_matrix((yVal,(yRow, yCol)),shape = (nS, nS))

    # normalize by the degree matrix to form normalized graph Laplacian:
    d = sum(l)
    d = d.toarray()
    d = np.sqrt(d)
    d = d.T
    yVal = yVal / (d[yRow].flatten('C') * d[yCol].flatten('C'))
    l = csc_matrix((yVal,(yRow, yCol)),shape = (nS, nS))
    l = np.abs( l + l.T ) / 2.0 #iron out numerical wrinkles
    temp = l-l.T

    return (l, sigmaTune)
