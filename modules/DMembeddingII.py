import numpy as np, logging
from scipy.sparse import csr_matrix
import time
import fergusonE
import sembeddingonFly
from collections import namedtuple
import p

'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2019 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)    
  
'''
def get_yColVal(params):

    yVal  = params[0]
    yVal1  = params[1]
    yCol  = params[2]
    yInd1  = params[3]
    nB = params[4]
    nN = params[5]
    nNIn = params[6]
    jStart = params[7]
    jEnd = params[8]
    indStart = params[9]
    indEnd = params[10]
    iBatch = params[11]

    DataBatch = yVal1
    DataBatch = DataBatch.reshape(nB, nNIn).T
    DataBatch = DataBatch[:nN, :]
    DataBatch[0,:] = 0
    yVal[indStart:indEnd] = DataBatch.reshape(nN * nB,1)
    DataBatch = yInd1
    DataBatch = DataBatch.reshape(nB, nNIn).T
    DataBatch = DataBatch[:nN, :]
    yCol[indStart:indEnd] = DataBatch.reshape(nN * nB,1).astype(float)

    return (yCol,yVal)

def initialize(nS, nN, D):
    yInd1 = np.zeros((nN,nS), dtype = 'int32')
    yVal1 = np.zeros((nN,nS), dtype='float64')

    for iS in range(nS):
        D[iS,iS] = -np.Inf # force this distance to be the minimal value
        B = np.sort(D[:,iS])
        IX = np.argsort(D[:,iS])
        yInd1[:,iS] = IX[:nN]
        yVal1[:,iS] = B[:nN]
        yVal1[0,iS] = 0  # set this distance back to zero

    yInd1 = yInd1.flatten('F')
    yVal1 = yVal1.flatten('F')
    return (yInd1,yVal1)

'''def construct_matrix(yVal, nS):
    ifZero = yVal < 1e-6
    yRowNZ = yRow[~ifZero]
    yColNZ = yCol[~ifZero]
    yValNZ = np.sqrt( yVal[~ifZer])
    y = csr_matrix((yValNZ,(yRowNZ, yColNZ )), shape=(nS, nS))
    # clear yRowNZ yColNZ yValNZ
    y2 = np.dot(y, y.T) # y2 contains the squares of the distances
    y = y**2
    y = y + y.T - y2
    # clear y2 % preserve memory
    return (y,nNZ,ifZero)
'''

def construct_matrix0(Row, Col, Val, nZ, nS):
    y = csr_matrix((Val, (Row, Col)), shape=(nS, nS)).toarray()
    y2 = y * y.T #y2 contains the squares of the distances
    y = y**2
    y = y + y.T - y2
    return y

def construct_matrix1(Row, Col, Val, nZ, nS):
    y = csr_matrix((Val, (Row, Col)), shape=(nS, nS)).toarray()
    y2 = y * y.T #y2 contains the squares of the distances
    y = y + y.T - y2
    return y

def op(D, k, tune, prefsigma): #*arg
    nS = D.shape[0]
    nN = k #total number of entries is nS*nN
    yInd1, yVal1 = initialize(nS, nN, D)

    # diffraction patterns:
    nB = nS #batch size (number of diff. patterns per batch)
    nNIn = k #number of input nearest neighbors
    nN = k #number of output nearest neighbors
    iBatch = 1 #REVIEW THIS SECTION, absurd to keep this here... nB=nS then nBatch = nS/nB ...???

    yVal = np.zeros((nS * nN, 1))
    yCol = np.zeros((nS * nN, 1))

    nBatch = int(nS / nB)
    for iBatch in range(nBatch):
        # linear indices in the non-symmetric distance matrix (indStart, indEnd)
        indStart = iBatch * nB * nN
        indEnd = (iBatch + 1) * nB * nN
        # diffraction pattern indices (jStart, jEnd):
        jStart = iBatch * nB
        jEnd = (iBatch + 1) * nB
        params = (yVal, yVal1, yCol, yInd1, nB, nN, nNIn,
                  jStart, jEnd, indStart, indEnd, iBatch)
        yCol, yVal = get_yColVal(params)

    # symmetrizing the distance matrix:
    yRow = np.ones((nN, 1)) * range(nS)
    yRow = yRow.reshape(nS * nN, 1)
    ifZero = yVal < 1e-6
    yRowNZ = yRow[~ifZero]
    yColNZ = yCol[~ifZero]
    yValNZ = np.sqrt(yVal[~ifZero])
    nNZ = len(yRowNZ) #number of nonzero elements in the non-sym matrix
    yRow = yRow[ifZero]
    yCol = yCol[ifZero]
    nZ = len(yRow) #number of zero elements in the non-sym matrix

    y = construct_matrix0(yRowNZ, yColNZ, yValNZ, nZ, nS)
    yRowNZ = y.nonzero()[0]
    yColNZ = y.nonzero()[1]
    yValNZ = y[y.nonzero()]
    nNZ = len(y.nonzero()[0]) #number of nonzero elements in the sym matrix

    y = construct_matrix1(yRow, yCol, np.ones((nZ,1)).flatten(), nZ, nS)
    y = csr_matrix((np.ones((nZ, 1)).flatten(), (yRow, yCol)), shape=(nS, nS)).toarray()
    yRow = y.nonzero()[0]
    yCol = y.nonzero()[1]
    yVal = y[y.nonzero()]

    yVal[:] = 0
    nZ = len(y.nonzero()[0]) #number of zero elements in the sym matrix
    yRow = np.hstack((yRow, yRowNZ)).astype(int)
    yCol = np.hstack((yCol, yColNZ)).astype(int)
    yVal = np.hstack((yVal, yValNZ))

    a0 = (np.random.rand(4, 1) - .5)
    count = 0
    resnorm = np.inf

    logEps = np.arange(-150, 150.2, 0.2)
    popt, logSumWij, resnorm, R_squared = fergusonE.op(np.sqrt(yVal), logEps, a0)
    nS = D.shape[0]
    nEigs = min(p.num_eigs, nS-3) #number of eigenfunctions to compute
    nA = 0 #autotuning parameter
    nN = k #number of nearest neighbors
    nNA = 0 #number of nearest neighbors used for autotuning
    if count < 20:
        alpha = 1 #kernel normalization
            #alpha = 1.0: Laplace-Beltrami operator
            #alpha = 0.5: Fokker-Planck diffusion
            #alpha = 0.0: graph Laplacian normalization
        sigma = tune * np.sqrt(2 * np.exp(-popt[1] / popt[0])) #Gaussian Kernel width (=1 for autotuning)
        #sigma = np.sqrt(2 * np.exp(-popt[1] / popt[0])) #as in Fergusson paper
    else:
        print('using prefsigma...') #does this ever happen or can we delete? REVIEW
        sigma = prefsigma
        alpha = 1

    visual = 1
    options = namedtuple('Options', 'sigma alpha visual nEigs')
    options.sigma = sigma
    options.alpha = alpha
    options.visual = visual
    options.nEigs = nEigs

    lamb, v = sembeddingonFly.op(yVal, yCol, yRow, nS, options)

    #psi = v[:, 1 : nEigs+1]/np.tile(v[:, 0 ].reshape((-1,1)), (1, nEigs))
    true_shape = v.shape[1] - 1
    psi = np.zeros((v.shape[0], nEigs))
    psi[:,:true_shape] = v[:, 1 :] / np.tile(v[:, 0 ].reshape((-1, 1)), (1, true_shape)) # could be fewer than nEigs

    ##################################
    # the Riemannian measure. Nov 2012
    mu = v[:, 0]
    mu = mu * mu #note: sum(mu)=1
    ##################################
        
    return (lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared) 
