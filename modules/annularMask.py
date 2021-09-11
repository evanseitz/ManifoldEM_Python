'''
Copyright (c) Columbia University Hstau Liao 2019    
'''

"""function mask = annularMask(a,b,N,M)
% mask = annularMask(a,b,N,M)
% 
% returns a N x M matrix with an annular (donut) mask of inner
% radius a and outer radius b. Pixels outside the donut or inside the hole
% are filled with 0, and the rest with 1.
% 
% The circles with radii a and b are centered on pixel (N/2,M/2).
% 
% Programmed December 2007, modified by Peter Schwander December 2008 (Python version by Hstau Liao 2018)
% Copyright (c) Russell Fung 2007 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import numpy as np
def op(a,b,N,M):
    aSq = a*a
    bSq = b*b
    mask = np.zeros((N,M))
    for xx in range(N):
        xDist = xx-N/2+1
        xDistSq = xDist*xDist
        for yy in range(M):
            yDist = yy-M/2
            yDistSq = yDist*yDist
            rSq = xDistSq+yDistSq
            mask[xx,yy] = (rSq>=aSq)&(rSq<bSq)
    
    return mask
