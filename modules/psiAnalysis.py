import logging, sys
import numpy as np
import multiprocessing
import psiAnalysisParS2
import myio
from functools import partial
from contextlib import contextmanager
import set_params
import time,os
from subprocess import Popen
import p
from pyface.qt import QtGui, QtCore
os.environ['ETS_TOOLKIT'] = 'qt4'

'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Hstau Liao 2018 (python version)
Copyright (c) Evan Seitz 2019 (python version) 
'''


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def fileCheck(N):
    fin_PDs = np.zeros(shape=(N,p.num_psis), dtype=int) #zeros signify PD_psi entry not complete
    for root, dirs, files in os.walk(p.psi2_prog):
        for file in sorted(files):
            if not file.startswith('.'): #ignore hidden files
                fin_PD, fin_psi = file.split('_')
                fin_PDs[int(fin_PD),int(fin_psi)] = int(1)
    return fin_PDs

def divid(N,rc,fin_PDs):
    ll = []
    for prD in range(N):
        dist_file = '{}prD_{}'.format(p.dist_file, prD)
        psi_file = '{}prD_{}'.format(p.psi_file, prD)
        psi2_file = '{}prD_{}'.format(p.psi2_file, prD)
        EL_file = '{}prD_{}'.format(p.EL_file, prD)
        psinums = rc['psiNumsAll'][prD,:]
        senses = rc['sensesAll'][prD,:]
        psi_list = [] #list of incomplete psi values per PD
        for psi in range(len(psinums)):
            if fin_PDs[int(prD),int(psi)] == int(1):
                continue
            else:
                psi_list.append(psi)
        ll.append([dist_file, psi_file, psi2_file, EL_file, psinums, senses, prD, psi_list])
    return ll

def op(*argv):
    time.sleep(5)
    set_params.op(1)
    #set_params.op(-1)
    
    multiprocessing.set_start_method('fork', force=True)

    psiNumsAll = np.tile(np.array(range(p.num_psis)), (p.numberofJobs, 1))  # numberofJobs x num_psis
    sensesAll = np.tile(np.ones(p.num_psis), (p.numberofJobs, 1))  # numberofJobs x num_psis
    rc = {'psiNumsAll': psiNumsAll, 'sensesAll': sensesAll}

    if p.machinefile:
        print('using MPI with {} processes'.format(p.ncpu))
        Popen(["mpirun", "-n", str(p.ncpu), "-machinefile", str(p.machinefile),
            "python", "modules/psiAnalysis_mpi.py",str(p.proj_name)],close_fds=True)
        if argv:
            progress3 = argv[0]
            offset = 0
            while offset < p.num_psis*p.numberofJobs:
                fin_PDs = fileCheck(p.numberofJobs)  # array of finished PDs (0's are unfinished, 1's are finished)
                offset = np.count_nonzero(fin_PDs==1)
                progress3.emit(int((offset / float((p.numberofJobs)*p.num_psis)) * 100))
                time.sleep(5)
    else:
        print("Computing the NLSA snapshots...")
        isFull=0
        fin_PDs = fileCheck(p.numberofJobs)  # array of finished PDs (0's are unfinished, 1's are finished)
        input_data = divid(p.numberofJobs,rc,fin_PDs)

        if argv:
            progress3 = argv[0]
            offset = np.count_nonzero(fin_PDs==1)
            progress3.emit(int((offset / float((p.numberofJobs)*p.num_psis)) * 100))

        print("Processing {} projection directions.".format(len(input_data)))

        if p.ncpu == 1:  # avoids the multiprocessing package
            for i in range(len(input_data)):
                if argv: #for p.ncpu=1, progress3 update happens inside psiAnalysisParS2;
                    # however, same signal can't be sent if multiprocessing
                    psiAnalysisParS2.op(input_data[i], p.conOrderRange, p.trajName, isFull, p.num_psiTrunc, argv[0])
                else: #for non-GUI
                    psiAnalysisParS2.op(input_data[i], p.conOrderRange, p.trajName, isFull, p.num_psiTrunc)

        else:
            with poolcontext(processes=p.ncpu,maxtasksperchild=1) as pool:
                for i, _ in enumerate(pool.imap_unordered(partial(psiAnalysisParS2.op,
                                conOrderRange=p.conOrderRange,
                                traj_name=p.trajName,isFull=isFull,psiTrunc=p.num_psiTrunc), input_data),1):
                    if argv:
                        progress3 = argv[0]
                        fin_PDs = fileCheck(p.numberofJobs) #array of finished PDs (0's are unfinished, 1's are finished)
                        offset = np.count_nonzero(fin_PDs==1)
                        progress3.emit(int((offset / float((p.numberofJobs)*p.num_psis)) * 100))

                    time.sleep(0.05)

                pool.close()
                pool.join()

    set_params.op(0)
    return

if __name__ == '__main__':
    import p, os
    p.init()
    p.user_dir = '../'
    p.out_dir = os.path.join(p.user_dir, 'data_output/')
    p.tess_file = '{}/selecGCs'.format(p.out_dir)
    p.nowTime_file = os.path.join(p.user_dir, 'data_output/nowTime')
    p.create_dir()
    op()
