import logging, sys
import numpy as np
import myio
import h5py,scipy
from scipy.io import loadmat
import writeRelionS2
import os.path
from subprocess import call
import gc
import set_params
import ComputeEnergy1D
import time,os,glob
from pyface.qt import QtGui, QtCore
os.environ['ETS_TOOLKIT'] = 'qt4'


''' %Version V 1.2
    % Copyright (c) UWM, Ali Dashti 2016 (matlab version)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %This script prepares the image stacks and orientations for 3D reconstruction.
    Copyright (c) Columbia Univ Hstau Liao 2018 (python version)   
    Copyright (c) Columbia University Suvrajit Maji 2020 (python version)    
'''

#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

def op(*argv):
    time.sleep(5)
    import p


    set_params.op(1)
    #ComputeEnergy1D.op() # this is s a repeat , commented out
    print("Writing output files...")


    
    data = myio.fin1(p.CC_file)
    psiNumsAll = data['psinums']

    range1 = np.arange(p.numberofJobs)
    # read reaction coordinates

    #psis, posPaths, imgLabelsAll, taus, listBads, ConOrders, inds = \
    #   registerVarsS2.op(psiNumsAll, sensesAll, p.remote_file, range1, p.conOrderRange, p.tau_file, p.num_ang)

    a = np.nonzero(psiNumsAll[0,:] == -1)[0] #unassigned states, python
    #print psiNumsAll.shape,'a=',a
    range = np.delete(range1, a)
    a = np.nonzero(p.trash_list == 1)[0]  # unassigned states, python
    range = np.delete(range,a)
    xSelect = range[0:5]

    # getFromFileS2
    xLost = []
    trajTaus = [None] * p.numberofJobs
    posPathAll= [None] * p.numberofJobs
    posPsi1All = [None] * p.numberofJobs

    for x in xSelect:
        EL_file = '{}prD_{}'.format(p.EL_file, x)
        File = '{}_{}_{}'.format(EL_file,p.trajName,1)
        print('var x',x)
        #print File
        if os.path.exists(File):
            data = myio.fin1(File)
            trajTaus[x] = data['tau']
            posPathAll[x] = data['posPath']
            posPsi1All[x] = data['PosPsi1']
        else:
            xLost.append(x)
            continue

    xSelect = list(set(xSelect)-set(xLost))

    # Section II
    visual = 0
    #trajTauOut_, sense_, start_ = trajTausAlign.op(trajTaus, xSelect, p.isTrajClosed, visual)
    tauAvg = np.array([])
    for x in xSelect:
        tau = trajTaus[x]
        tau = tau.flatten()
        tau = (tau - np.min(tau)) / (np.max(tau) - np.min(tau))
        tauAvg = np.concatenate((tauAvg,tau.flatten()))


    ## added June 2020, S.M.
    traj_file2 = "{}name{}_vars".format(p.traj_file,p.trajName)
    myio.fout1(traj_file2, ['trajTaus', 'posPsi1All', 'posPathAll', 'xSelect', 'tauAvg'], [trajTaus, posPsi1All, posPathAll, xSelect, tauAvg])
    gc.collect()


    # Section III
    if argv:
        writeRelionS2.op(trajTaus, posPsi1All, posPathAll, xSelect, tauAvg, argv[0])

    else:
        writeRelionS2.op(trajTaus, posPsi1All, posPathAll, xSelect, tauAvg)

    gc.collect()
    if argv:
        progress7 = argv[0]
        progress7.emit(100)

    print('finished manifold embedding!')

if __name__ == '__main__':
    import p, sys, os
    sys.path.append('../')
    p.user_dir = '../'
    p.out_dir = os.path.join(p.user_dir, 'data_output/')
    p.tess_file = '{}/selecGCs'.format(p.out_dir)
    p.init()
    p.create_dir()
    op()
