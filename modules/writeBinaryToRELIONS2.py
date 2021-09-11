import logging, sys
import numpy as np
import mrcfile
import q2Spider
import os.path
import star
import pandas

import time,os
from pyface.qt import QtGui, QtCore
os.environ['ETS_TOOLKIT'] = 'qt4'

'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
Copyright (c) Columbia University Sonya Hanson 2018 (python version)    
'''

#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

def flip1(data):
    N,dim,dim = data.shape
    for i in range(N):
        img = data[i,:,:]
        data[i,:,:] = img.T
    return data

def op(nClass,nPix,trajName,relion_dir, bin_dir,*argv):    # Orientations

    for j in range(nClass):
        f3 = '{}ProjDirTraj{}_{}_of_{}.dat'.format(bin_dir, trajName, j+1, nClass)
        if os.path.exists(f3):
            GCs = np.array(np.fromfile(f3,dtype=int))

            f4 = '{}quatsTraj{}_{}_of_{}.dat'.format(bin_dir, trajName, j+1, nClass)
            qs = np.array(np.fromfile(f4,dtype=float))
            #qs = qs.reshape(4, len(GCs))
            qs = qs.reshape(len(GCs),4)
            qs = qs.T

            #print 'qs=',np.shape(qs),len(GCs)
            PDs = 2 * np.vstack((qs[1,:]*qs[3,:] - qs[0,:]*qs[2,:],
                                qs[0,:]*qs[1,:] + qs[2,:]*qs[3,:],
                                qs[0,:]**2 + qs[3,:]**2 - np.ones((1, len(GCs))) / 2))
            phi = np.empty(len(GCs))
            theta = np.empty(len(GCs))
            psi = np.empty(len(GCs))

            for i  in range(len(GCs)):
                PD = PDs[:, i]
                lPD = sum(PD**2)
                Qr = np.array([1 + PD[2], PD[1], -PD[0], 0])
                #print i, j, Qr,GCs
                Qr = Qr / np.sqrt(np.sum(Qr**2))
                #Qr = np.array([0.6807, -0.7263, .0951, 0])
                phi[i],theta[i],psi[i] = q2Spider.op(Qr)
                
            phi = np.mod(phi,2*np.pi)*(180/np.pi)
            theta = np.mod(theta,2*np.pi)*(180/np.pi)
            psi = 0.0 #np.mod(psi,2*np.pi)*(180/np.pi) already done in getDistance

        f1 = '{}NLSAImageTraj{}_{}_of_{}.dat'.format(bin_dir, trajName,j+1, nClass)
        if os.path.exists(f1):
            data = np.array(np.fromfile(f1)) #, dtype=np.float32))
            data= data.astype(np.float32)

            n = data.shape[0] / nPix**2
            #print 'data',np.shape(data),n,nPix
            data = data.reshape(n, nPix, nPix) # flip here
            data = flip1(data) # flip here
            #print 'data-reshape',np.shape(data)
            traj_file = '{}imgsRELION_{}_{}_of_{}.mrcs'.format(relion_dir, trajName, j + 1, nClass)
            if os.path.exists(traj_file):
                mrc = mrcfile.open(traj_file,'r+')
            else:
                mrc = mrcfile.new(traj_file)

            #mrc.set_data(data*-1) #*-1 inverts contrast
            mrc.set_data(data*-1)

            ang_file = '{}EulerAngles_{}_{}_of_{}.star'.format(relion_dir,trajName,j+1,nClass)
            d = dict(phi=phi, theta=theta,psi=psi)
            df = pandas.DataFrame(data=d)
            # should be relative path:
            traj_file = 'imgsRELION_{}_{}_of_{}.mrcs'.format(trajName, j + 1, nClass)
            star.write_star(ang_file,traj_file,df)

        if argv:
            progress7 = argv[0]
            signal = 50 + int((j / float(nClass)) * 50)
            if signal == 100:
                signal = 95
            progress7.emit(signal)


    res = 'ok'
    return res
