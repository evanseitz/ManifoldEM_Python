'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
'''

import numpy as np

def op(CTF, posPath, posPsi1, ConOrder, num):
    dim = CTF.shape[1]
    SNR = 5
    CTF1 = CTF[posPath[posPsi1], :, :]
    wiener_dom = np.zeros((num - ConOrder, dim, dim), dtype='float64')
    for i in range(num - ConOrder):
        for ii in range(ConOrder):
            ind_CTF = ConOrder - ii + i
            wiener_dom[i, :, :] = wiener_dom[i, :, :] + CTF1[ind_CTF, :, :] ** 2

    wiener_dom = wiener_dom + 1. / SNR

    return (wiener_dom,CTF1)

