import multiprocessing
import psiAnalysisParS2
import myio
from functools import partial
from contextlib import contextmanager
import set_params
import time,os
import p
import numpy as np
import ComputeEnergy1D
from subprocess import Popen
from pyface.qt import QtGui, QtCore
os.environ['ETS_TOOLKIT'] = 'qt4'
'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def fileCheck():
    fin_PDs = [] #collect list of previously finished PDs from ELConc{}/
    for root, dirs, files in os.walk(p.EL_prog):
        for file in sorted(files):
            if not file.startswith('.'): #ignore hidden files
                fin_PDs.append(int(file))
    return fin_PDs

def divide1(R,psiNumsAll,sensesAll):
    ll = []
    fin_PDs = fileCheck()
    for prD in R:
        dist_file = '{}prD_{}'.format(p.dist_file, prD)
        psi_file = '{}prD_{}'.format(p.psi_file, prD)
        psi2_file = '{}prD_{}'.format(p.psi2_file,prD)
        EL_file = '{}prD_{}'.format(p.EL_file, prD)
        psinums = [psiNumsAll[0,prD]]
        senses = [sensesAll[0,prD]]
        if prD not in fin_PDs:
            ll.append([dist_file, psi_file, psi2_file, EL_file, psinums, senses, prD])

    return ll

def count1(R):
    c = len(R) - len(fileCheck())
    return c

def op(*argv):
    time.sleep(5)
    set_params.op(1)
    #set_params.op(-1)
    
    multiprocessing.set_start_method('fork', force=True)
    
    R = np.array(range(p.numberofJobs))
    R = np.delete(R,np.nonzero(p.trash_list==1)[0])
    if p.machinefile:
        print('using MPI with {} processes'.format(p.ncpu))
        Popen(["mpirun", "-n", str(p.ncpu), "-machinefile", str(p.machinefile),
            "python", "modules/EL1D_mpi.py",str(p.proj_name)],close_fds=True)
        if argv:
            progress6 = argv[0]
            offset = 0
            while offset < len(R):
                offset = len(R) - count1(R)
                progress6.emit(int((offset / float(len(R))) * 100))
                time.sleep(5)
    else:
        print("Recomputing the NLSA snapshots using the found reaction coordinates...")
        data = myio.fin1(p.CC_file)
        psiNumsAll = data['psinums']
        sensesAll = data['senses']
        isFull=1
        input_data = divide1(R,psiNumsAll,sensesAll)
        if argv:
            progress6 = argv[0]
            offset = len(R) - len(input_data)
            progress6.emit(int((offset / float(len(R))) * 100))

        print("Processing {} projection directions.".format(len(input_data)))

        if p.ncpu == 1:  # avoids the multiprocessing package
            for i in range(len(input_data)):
                psiAnalysisParS2.op(input_data[i], p.conOrderRange, p.trajName, isFull,p.num_psiTrunc)
                if argv:
                    offset += 1
                    progress6.emit(int((offset / float(len(R))) * 100))
        else:
            with poolcontext(processes=p.ncpu,maxtasksperchild=1) as pool:
                for i, _ in enumerate(pool.imap_unordered(partial(psiAnalysisParS2.op,conOrderRange=p.conOrderRange,
                                       traj_name=p.trajName,isFull=isFull,psiTrunc=p.num_psiTrunc), input_data),1):
                    if argv:
                        offset += 1
                        progress6.emit(int((offset / float(len(R))) * 100))
                    time.sleep(0.05)
                pool.close()
                pool.join()

    ComputeEnergy1D.op()
    set_params.op(0)
    return

if __name__ == '__main__':
    import p, os
    print("Recomputing the NLSA snapshots using the found reaction coordinates...")
    p.init()
    p.user_dir = '../'
    p.out_dir = os.path.join(p.user_dir, 'data_output/')
    p.tess_file = '{}/selecGCs'.format(p.out_dir)
    p.nowTime_file = os.path.join(p.user_dir, 'data_output/nowTime')
    p.create_dir()

    op()
