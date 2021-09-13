import logging, sys
import numpy as np
import myio
import util
import time,os
import quaternion
import mrcfile
import pandas
import star
from pyface.qt import QtGui, QtCore
os.environ['ETS_TOOLKIT'] = 'qt4'
import matplotlib.pyplot as plt
from scipy import io
import copy
'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Hstau Liao 2018 (python version)   
Copyright (c) Columbia University Suvrajit Maji 2020 (python version) 
'''

def flip1(data):
    N,dim,dim = data.shape
    for i in range(N):
        img = data[i,:,:]
        data[i,:,:] = img.T
    return data

#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

def op(trajTaus, posPsi1All, posPathAll, xSelect, tauAvg, *argv):
    import p
    
    # S.M. June 2020
    i = 0
    pathw = p.width_1D
    #print('pathw',pathw)
    
    # TO DO: Have to find a way to control this from the GUI and also write extra steps to provide resume capability
    get_traj_bins=1 # if 1, then the trajectory data is extracted from selected PDs, 
                    #if 0 then we skip this and read from previously saved files
    numberOfWorkers = 20 # determines how many PDs will be processed and saved together in a single pickle file
                         # higer values mean overall fewer files written, but will also need more memory to write and read later.
    numberOfJobs=len(xSelect)
    
    # S.M. June 2020
    if get_traj_bins:


        print('Extracting and writing individual trajectory data from selected projection directions ...')

        for num in range(0,numberOfJobs,numberOfWorkers):
            #print('\nnum',num)

            imgss = [[]]*p.nClass
            phis = [[]]*p.nClass
            thetas = [[]] * p.nClass
            psis = [[]] * p.nClass


            # somehow this step is necessary
            for bin in range(p.nClass):
                imgss[bin] = []
                thetas[bin] = []
                phis[bin] = []
                psis[bin] = []

            numNext  = min(numberOfJobs,num+numberOfWorkers)

            idx = np.arange(num,numNext)
            xSel = np.array(xSelect)[idx]
            for x in xSel:
                #print('bx=',x)
                i += 1
                EL_file = '{}prD_{}'.format(p.EL_file, x)
                File = '{}_{}_{}'.format(EL_file, p.trajName, 1)
                data = myio.fin1(File)

                IMGT = data['IMGT']

                posPath = posPathAll[x]
                psi1Path = posPsi1All[x]

                dist_file = '{}prD_{}'.format(p.dist_file, x)
                data = myio.fin1(dist_file)
                q = data['q']

                q = q[:, posPath[psi1Path]] # python
                nS = q.shape[1]

                conOrder = np.floor(float(nS)/p.conOrderRange).astype(int)
                copies = conOrder
                q = q[:,copies-1:nS-conOrder]

                IMGT = IMGT / conOrder
                IMGT = IMGT.T  #flip here IMGT is now num_images x dim^2

                tau = trajTaus[x]
                tauEq = util.hist_match(tau, tauAvg)

                #import matplotlib.pyplot as plt
                #plt.hist(tauEq,50)
                #plt.show()
                #del data # june 2020

                IMG1 = np.zeros((p.nClass, IMGT.shape[1]))


                for bin in range(p.nClass - pathw + 1):
                    #print 'bin is', bin
                    if bin == p.nClass - pathw:
                        tauBin = ((tauEq >= ((bin + float(0.0)) / p.nClass)) & (tauEq <= (bin + float(pathw)) / p.nClass)).nonzero()[0]
                    else:
                        tauBin = ((tauEq >= ((bin + float(0.0)) / p.nClass)) & (tauEq < (bin + float(pathw)) / p.nClass)).nonzero()[0]

                    #print 'lb',bin / float(p.nClass), 'ub',(bin + pathw) / float(p.nClass)


                    if len(tauBin) == 0:
                        #print 'bad bin is',bin
                        continue
                    else:
                        imgs = IMGT[tauBin,:].astype(np.float32)
                        ar2 = tauEq[tauBin]
                        qs = q[:,tauBin]
                        nT = len(tauBin)
                        PDs = quaternion.calc_avg_pd(qs,nT)
                        phi = np.empty(nT)
                        theta = np.empty(nT)
                        psi = np.empty(nT)

                        for i in range(nT):
                            PD = PDs[:, i]
                            phi[i], theta[i], psi[i] = quaternion.psi_ang(PD)
                        dim = int(np.sqrt(imgs.shape[1]))
                        imgs = imgs.reshape(nT, dim, dim)  # flip here
                        imgs = flip1(imgs)  # flip here

                        imgss[bin].append(imgs)  # append here
                        phis[bin].append(phi)
                        thetas[bin].append(theta)
                        psis[bin].append(psi)

                        #print('nT',nT,'tauBin shape',np.shape(tauBin),tauBin)
                        # june 2020ss
                        #del qs,PDs
                        #del phi,theta,psi #,imgs

                        #print 'imgs shape',np.shape(imgs),'imgss at bin=',bin,', x=',x,np.shape(imgss[bin][np.mod(x,numberOfWorkers)]) # in groups of numWorkers and then the array is reset and used for appending

                #del q #,IMT# june 2020

            #print 'imgss at bin',bin,np.shape(imgss[bin]),'\n'

            traj_bin_file = "{}name{}_group_{}_{}".format(p.traj_file,p.trajName,num,numNext-1)
            key_list=['imgss','phis','thetas','psis']
            v_list = [imgss,phis,thetas,psis]
            myio.fout1(traj_bin_file,key_list,v_list)
            time.sleep(10)
            print('Done saving group.')

    else:
        print('Reading previously generated trajectory data from saved projection directions...')

    # S.M. June 2020
    # loop through the nClass again and convert each list in the list to array
    for bin in range(42, p.nClass - pathw + 1):
        print('\nConcatenated bin:',bin)


        for num in range(0,numberOfJobs,numberOfWorkers):

            #print 'num',num
            numNext  = min(numberOfJobs,num+numberOfWorkers)

            traj_bin_file = "{}name{}_group_{}_{}".format(p.traj_file,p.trajName,num,numNext-1)

            data = myio.fin1(traj_bin_file)
            imgss_bin_g = data['imgss']
            phis_bin_g = data['phis']
            thetas_bin_g = data['thetas']
            psis_bin_g = data['psis']

            data =[]
            del data

            for x in range(num,numNext):
                y = np.mod(x,numberOfWorkers)
                #print 'x=',x,'y=',y,'imgss_bin_g shape',np.shape(imgss_bin_g[bin])[0]
                if y >=np.shape(imgss_bin_g[bin])[0]:
                    continue
                #print 'imgss_bin_g', np.shape(imgss_bin_g[bin][y])

                if num==0 and x==0:
                    imgs = copy.deepcopy(imgss_bin_g[bin][y])
                    phi  = copy.deepcopy(phis_bin_g[bin][y])
                    theta = copy.deepcopy(thetas_bin_g[bin][y])
                    psi = copy.deepcopy(psis_bin_g[bin][y])
                else:
                    # reuse var names


                    imgs = np.concatenate([imgs, imgss_bin_g[bin][y]])
                    phi  = np.concatenate([phi, phis_bin_g[bin][y]])
                    theta = np.concatenate([theta, thetas_bin_g[bin][y]])
                    psi = np.concatenate([psi, psis_bin_g[bin][y]])
                #print 'imgs',np.shape(imgs)

        if len(imgss_bin_g[bin]) == 0:
            print('Bad bin:',bin)
            continue

        # reset , clear variables from workspace
        imgss_bin_g =[]
        phis_bin_g =[]
        thetas_bin_g =[]
        psis_bin_g=[]
        del imgss_bin_g , phis_bin_g , thetas_bin_g , psis_bin_g

        print('Concatenated imgs, shape',np.shape(imgs))
        #print imgs[0,0:5,0:5]
        #print imgs[7,0:5,0:5]

        #print 'phi',np.shape(phi)
        #print 'phi', phi
        #print 'theta',theta
        #print 'psi',psi


        print('Start writing trajectory data to file...'.format(bin))


        # print out
        traj_file_rel = 'imgsRELION_{}_{}_of_{}.mrcs'.format(p.trajName, bin + 1, p.nClass)
        traj_file = '{}{}'.format(p.relion_dir, traj_file_rel)
        ang_file = '{}EulerAngles_{}_{}_of_{}.star'.format(p.relion_dir, p.trajName, bin + 1, p.nClass)

        if os.path.exists(traj_file):
            mrc = mrcfile.open(traj_file, mode='r+')
        else:
            mrc = mrcfile.new(traj_file)
            # mrc.set_data(data*-1) #*-1 inverts contrast
        mrc.set_data(imgs * -1)
        #mrc.flush()
        time.sleep(5)
        mrc.close()

        d = dict(phi=phi, theta=theta, psi=psi)
        df = pandas.DataFrame(data=d)
        star.write_star(ang_file, traj_file_rel, df)
        print('Writing trajectory data for bin {} into .mrcs and .star format... done.\n'.format(bin))

    if argv:
        progress7 = argv[0]
        signal = int((bin / float(p.nClass)) * 100)
        if signal == 100:
            signal = 95
        progress7.emit(signal)
    res = 'ok'
    return res

