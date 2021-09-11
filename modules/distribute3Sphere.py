""" function [results, iter] = distribute3Sphere(numPts)
% [results, iter] = distribute3Sphere(numPts)
% distributes numPts points roughly uniformly on a unit 3-sphere and
% returns the coordinates in results. Number of iterations required is
% returned in iter.
% 
% Algorithm adapted from L. Lovisolo and E.A.B. da Silva, Uniform
% distribution of points on a hyper-sphere with applications to vector
% bit-plane encoding, IEE Proc.-Vis. Image Signal Process., Vol. 148, No.
% 3, June 2001
% 
% Programmed February 2009
% Copyright (c) Russell Fung 2009
% 
% Copyright (c) Columbia University Hstau Liao 2018 (python version)    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import numpy as np
import math
def op(numPts):
    maxIter = 100
    K = numPts
    A3 = 4*np.pi # surface area of a unit 3-sphere
    delta = math.exp(math.log(A3/K)/2.)
    results = np.zeros((2*K,3)); # algorithm sometimes returns more/ less points
    iter = 0
    id = 0

    while id!=K and iter<maxIter:
        iter = iter+1
        id = 0
        dw1 = delta
        for w1 in np.arange(0.5*dw1,np.pi,dw1):
            cw1 = math.cos(w1)
            sw1 = math.sin(w1)
            x1 = cw1
            dw2 = dw1/sw1
            for w2 in np.arange(0.5*dw2,2*np.pi,dw2):
                cw2 = math.cos(w2)
                sw2 = math.sin(w2)
                x2 = sw1*cw2
                x3 = sw1*sw2
            
                results[id,:] = np.hstack((x1, x2, x3))
                id = id + 1

        delta = delta*math.exp(math.log(float(id)/K)/2.)

    results = results[0:K,:]
    return (results,iter)
