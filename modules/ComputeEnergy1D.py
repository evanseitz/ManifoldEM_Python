import logging, sys
import numpy as np
import myio
import h5py,scipy
from scipy.io import loadmat
import writeBinaryImagesS2
import writeBinaryToRELIONS2
import p
import os.path
from subprocess import call
import gc
import set_params
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
    print("Computing the energy landscape...")
    set_params.op(1)

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
    xSelect = range
    #print 'range1=',range1
    #print 'a=',a
    #print 'select=',xSelect
    # getFromFileS2
    xLost = []
    trajTaus = [None] * p.numberofJobs
    posPathAll= [None] * p.numberofJobs
    posPsi1All = [None] * p.numberofJobs

    #print 'x loop 1'
    for x in xSelect:
        #print('x1:',x)
        EL_file = '{}prD_{}'.format(p.EL_file, x)
        File = '{}_{}_{}'.format(EL_file,p.trajName,1)

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
    hUn = np.zeros((1, p.nClass)).flatten()
    tauAvg = np.array([])

    #print 'x loop 2'
    for x in xSelect:
        #print('x2:',x)
        tau = trajTaus[x]
        tau = tau.flatten()
        #print 'tau1',tau
        tau = (tau - np.min(tau)) / (np.max(tau) - np.min(tau))
        #xBins = np.arange(1./(2.*p.nClass), 1+1./p.nClass, 1./p.nClass)
        #bins = np.linspace(0, 1, p.nClass+1)
        #bins = np.arange(1./(p.nClass*2), 1., 1./p.nClass)
        h,ctrs = np.histogram(tau,p.nClass)
        hUn = hUn + h
        tauAvg = np.concatenate((tauAvg,tau.flatten()))

    # Section III
    traj_file = "{}name{}".format(p.traj_file,p.trajName)
    myio.fout1(traj_file, ['hUn'], [hUn])

    #added June 2020, S.M.
    p.traj_file_vars = "{}name{}_vars".format(p.traj_file,p.trajName)
    myio.fout1(p.traj_file_vars, ['trajTaus', 'posPsi1All', 'posPathAll', 'xSelect', 'tauAvg'], [trajTaus, posPsi1All, posPathAll, xSelect, tauAvg])
    gc.collect()

    if argv:
        progress7 = argv[0]
        progress7.emit(100)

    p.hUn = hUn
    OM_file = '{}OM'.format(p.OM_file)
    hUn.astype('int').tofile(OM_file)

    #################
    # compute energy:
    T = p.temperature  # Celsius, may need to be user-defined
    kB = 0.0019872041  # Boltzmann constant kcal / Mol / K
    rho = np.fmax(hUn, 1)
    kT = kB * (T + 273.15)  # Kelvin
    E = -kT * np.log(rho)
    E = E - np.amin(E)  # shift so that lowest energy is zero
    OM1_file = '{}EL'.format(p.OM1_file)
    E.astype('float').tofile(OM1_file)

    set_params.op(0)
    return hUn

if __name__ == '__main__':
    import p, sys, os
    sys.path.append('../')
    p.user_dir = '../'
    p.out_dir = os.path.join(p.user_dir, 'data_output/')
    p.tess_file = '{}/selecGCs'.format(p.out_dir)
    p.init()
    p.create_dir()
    op()
