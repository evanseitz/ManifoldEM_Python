import numpy as np, logging
import DMembeddingII
import pickle
import svdRF
import sys
import fit_1D_open_manifold_3D
import L2_distance
from collections import namedtuple
import get_wiener
from scipy.fftpack import fft2
from scipy.fftpack import ifft2
import myio
import gc
import time
'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
Copyright (c) Columbia University Suvrajit Maji 2020 (python version)
'''
#def op(NLSAPar, DD, posPath, posPsi1, imgAll, CTF, ExtPar):
def op(NLSAPar, DD, posPath, posPsi1, imgAll, msk2, CTF, ExtPar): #pass the msk2 var also
    num = NLSAPar['num']
    ConOrder = NLSAPar['ConOrder']
    k = NLSAPar['k']
    tune = NLSAPar['tune']
    nS = NLSAPar['nS']
    psiTrunc = NLSAPar['psiTrunc']

    ConD = np.zeros((num - ConOrder, num - ConOrder))
    for i in range(ConOrder):
        Ind = range(i,num - ConOrder + i)
        ConD += DD[Ind][:, Ind]
    '''
    for iii in range(num - ConOrder):
        for ii in range(num - ConOrder):
            for ConNum in range(ConOrder):
                Ind1 = iii + ConNum
                Ind2 = ii + ConNum
                ConD[iii, ii] += DD[Ind1, Ind2]
    '''

    # find the manifold mapping:
    lambdaC, psiC, sigmaC, mu, logEps, logSumWij, popt, R_squared = DMembeddingII.op(ConD, k, tune, 600000)  ### USE THE MU FROM SUPERVECTORS' DISTANCES

    lambdaC = lambdaC[lambdaC > 0]  ## lambdaC not used? REVIEW
    psiC1 = np.copy(psiC)
    # rearrange arrays
    if 'prD' in ExtPar:
        IMG1 = imgAll[posPath[posPsi1], :, :]
        # Wiener filtering
        wiener_dom, CTF1 = get_wiener.op(CTF, posPath, posPsi1, ConOrder, num)
    elif 'cuti' in ExtPar:
        IMG1 = imgAll[posPsi1, :, :]

    dim = CTF.shape[1]
    sigma = sigmaC
    ell = psiTrunc - 1
    N = psiC.shape[0]
    psiC = np.hstack((np.ones((N, 1)), psiC[:, 0:ell]))
    mu_psi = mu.reshape((-1, 1)) * psiC
    A = np.zeros((ConOrder * dim * dim, ell + 1), dtype='float64')
    tmp = np.zeros((dim * dim, num - ConOrder), dtype='float64')

    for ii in range(ConOrder):
        for i in range(num - ConOrder):
            ind1 = 0
            ind2 = dim * dim #max(IMG1.shape)
            ind3 = ConOrder - ii + i - 1
            img = IMG1[ind3, :, :]
            if 'prD' in ExtPar:
                img_f = fft2(img)#.reshape(dim, dim)) T only for matlab
                CTF_i = CTF1[ind3, :, :]
                #wiener_dom_i = wiener_dom[:, i].reshape(dim, dim)
                img_f_wiener = img_f * (CTF_i / wiener_dom[i, :, :])
                img = ifft2(img_f_wiener).real
                #img = np.squeeze(ifft2(img_f_wiener).real.T.reshape(-1, 1))
                img = img*msk2 # April 2020
            tmp[ind1:ind2, i] = np.squeeze(img.T.reshape(-1, 1))
            #tmp[ind1:ind2, i] = np.squeeze(img.reshape(-1,1))

        mm = dim * dim #max(IMG1.shape)
        ind4 = ii * mm
        ind5 = ind4 + mm
        A[ind4:ind5, :] = np.matmul(tmp, mu_psi)

    TF = np.isreal(A)
    if TF.any() != True:
        print('A is an imaginary matrix!')
        sys.exit

    U, S, V = svdRF.op(A)
    VX = np.matmul(V.T, psiC.T)

    sdiag = np.diag(S)

    Npixel = dim * dim #max(IMG1.shape[0])
    Topo_mean = np.zeros((Npixel, psiTrunc))
    for ii in range(psiTrunc):  # of topos considered
        #s = s + 1  needed?
        Topo = np.ones((Npixel, ConOrder)) * np.Inf

        for k in range(ConOrder):
            Topo[:, k] = U[k * Npixel : (k + 1) * Npixel, ii]
        Topo_mean[:, ii] = np.mean(Topo, axis=1)

    # unwrapping... REVIEW; allow user option to select from a list of chronos ([0,1,3]) to retain (i.e., not just i1, i2)
    i2 = 1
    i1 = 0

    ConImgT = np.zeros((max(U.shape), ell + 1), dtype='float64')
    for i in range(i1, i2 + 1):
        # %ConImgT = U(:,i) *(sdiag(i)* V(:,i)')*psiC';
        ConImgT = ConImgT + np.matmul(U[:,i].reshape(-1, 1), sdiag[i] * (V[:,i].reshape(1, -1)))

    recNum = ConOrder
    #tmp = np.zeros((Npixel,num-ConOrder),dtype='float64')
    IMGT = np.zeros((Npixel, nS - ConOrder - recNum), dtype='float64')
    for i in range(recNum):
        ind1 = i * Npixel
        ind2 = ind1 + Npixel
        tmp = np.matmul(ConImgT[ind1:ind2, :], psiC.T)
        for ii in range(num - 2 * ConOrder):
            ind3 = i + ii
            ttmp = IMGT[:, ii]
            ttmp = ttmp+tmp[:, ind3]
            IMGT[:, ii] = ttmp

    # normalize per frame so that mean=0 std=1, whole frame (this needs justif)
    for i in range(IMGT.shape[1]):
        ttmp = IMGT[:, i]
        try:
            ttmp = (ttmp - np.mean(ttmp)) / np.std(ttmp)
        except:
            print("flat image")
            exit(0)
        IMGT[:, i] = ttmp


    nSrecon = min(IMGT.shape)
    #dim = max(IMGT.shape)
    Drecon = L2_distance.op(IMGT, IMGT)
    k = nSrecon

    lamb, psirec, sigma, mu, logEps, logSumWij, popt, R_squared = DMembeddingII.op((Drecon**2), k, tune, 30)

    lamb = lamb[lamb > 0]
    a, b, tau = fit_1D_open_manifold_3D.op(psirec)

    # tau is #part (num-2ConOrder?)
    # psirec is #part x #eigs

    if NLSAPar['save'] == True:
        myio.fout1(ExtPar['filename'], ['psirec', 'tau', 'a', 'b'], [psirec, tau, a, b])


    return (IMGT, Topo_mean, psirec, psiC1, sdiag, VX, mu, tau)

