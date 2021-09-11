import logging, sys
import numpy as np
import myio
import util

import time,os
from pyface.qt import QtGui, QtCore
os.environ['ETS_TOOLKIT'] = 'qt4'
import h5py

import matplotlib.pyplot as plt
'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Hstau Liao 2018 (python version)    
'''


#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

def op(trajTaus, posPsi1All, posPathAll, xSelect, tauAvg, *argv):
    import p
    i = 0
    for x in xSelect:
        #print 'x=',x
        i += 1
        EL_file = '{}prD_{}'.format(p.EL_file, x)
        File = '{}_{}_{}'.format(EL_file, p.trajName, 1)
        data = myio.fin1(File)

        IMGT = data['IMGT']
        #print 'prd=',x
        #print 'sum IMGT=',sum(sum(IMGT))
        #print 'IMGT=',np.shape(IMGT)

        posPath = posPathAll[x]
        psi1Path = posPsi1All[x]

        #print 'posPath',posPath
        #print 'psi1Path',psi1Path

        dist_file = '{}prD_{}'.format(p.dist_file, x)
        data = myio.fin1(dist_file)
        q = data['q']
        #print 'qo=',np.shape(q),q[:,range(0,5)]

        q = q[:, posPath[psi1Path]] # python
        nS = q.shape[1]
        #print 'qpos=',np.shape(q),q[:,range(0,5)]

        conOrder = np.floor(float(nS)/p.conOrderRange).astype(int)
        copies = conOrder
        q = q[:,copies-1:nS-conOrder]
        #print 'qcop=',np.shape(q),q[:,range(0,5)]

        IMGT = IMGT / conOrder
        IMGT = IMGT.T  #flip here IMGT is now num_images x dim^2
        #print 'IMGT.shape',np.shape(IMGT)


        tau = trajTaus[x]
        tauEq = util.hist_match(tau, tauAvg)
        #print 'tauEq',tauEq
        pathw = p.width_1D
        IMG1 = np.zeros((p.nClass,IMGT.shape[1])) 
        for bin in range(p.nClass-pathw + 1):
            #print 'bin is', bin
            if bin == p.nClass - pathw:
                tauBin = ((tauEq >= (bin / float(p.nClass))) & (tauEq <= (bin + pathw) / p.nClass)).nonzero()[0]
            else:
                tauBin = ((tauEq >= (bin / float(p.nClass))) & (tauEq < (bin + pathw) / p.nClass)).nonzero()[0]

            if len(tauBin) == 0:
                #print 'bad bin is',bin
                continue
            else:
                #print 'something',bin
                f1 = '{}NLSAImageTraj{}_{}_of_{}.dat'.format(p.bin_dir,p.trajName,bin+1,p.nClass)
                f2 = '{}TauValsTraj{}_{}_of_{}.dat'.format(p.bin_dir,p.trajName,bin+1,p.nClass)
                f3 = '{}ProjDirTraj{}_{}_of_{}.dat'.format(p.bin_dir,p.trajName,bin+1,p.nClass)
                f4 = '{}quatsTraj{}_{}_of_{}.dat'.format(p.bin_dir,p.trajName,bin+1,p.nClass)

                ar1 = IMGT[tauBin,:]

                #print "ar1:",np.shape(ar1)
                #print "tauBin:",np.shape(tauBin),tauBin
                with open(f1, "ab") as f:  # or choose 'wb' mode
                    ar1.astype('float').tofile(f)
                ar2 = tauEq[tauBin]
                with open(f2, "ab") as f:
                    ar2.astype('float').tofile(f)
                ar3 = x * np.ones((1, len(tauBin)))
                with open(f3, "ab") as f:
                    ar3.astype('int').tofile(f)
                ar4 = q[:,tauBin]
                #print 'ar4=',ar4
                ar4 = ar4.flatten('F')
                with open(f4, "ab") as f:
                    ar4.astype('float64').tofile(f)

                IMG1[bin, :] = np.mean(IMGT[tauBin,:], axis=0)

        bin_file = '{}PD_{}_Traj{}'.format(p.bin_dir,x,p.trajName)
        myio.fout1(bin_file, ['IMG1'], [IMG1])

        if argv:
            progress7 = argv[0]
            signal = int((i / float(len(xSelect))) * 50)
            progress7.emit(signal)

    res = 'ok'
    return res

