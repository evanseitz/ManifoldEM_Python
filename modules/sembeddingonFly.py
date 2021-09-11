from scipy.sparse.linalg import eigsh
from collections import namedtuple
import slaplacianonFly
import numpy as np
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

"""
%SEMBEDDING  Laplacian eigenfunction embedding using sparse arrays

"""
'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
'''

def op( yVal,yCol,yRow, nS, options1):
    options = namedtuple('options', 'sigma alpha visual nEigs autotune')
    options.sigma = options1.sigma
    options.alpha = options1.alpha
    options.nEigs = options1.nEigs
    options.autotune = 0

    l,sigmaTune = slaplacianonFly.op(yVal,yCol,yRow,nS, options)
    #print 'Embedding Eigs  = ' + str( options.nEigs ) + '\n'
    try:
        vals, vecs = eigsh(l, k=options.nEigs+1,maxiter=300)
    except ArpackNoConvergence as e:
        #print(e)
        vals = e.eigenvalues
        vecs = e.eigenvectors
        print("eigsh not converging in 300 iterations...") 
    ix = np.argsort(vals)[::-1]
    vals = np.sort(vals)[::-1]
    vecs = vecs[:,ix]
    #print 'sembed: vecs is', vecs[10:20,10:20]
    #print 'sembed: vals is', vals[-1:-10:-1]

    return (vals, vecs)

