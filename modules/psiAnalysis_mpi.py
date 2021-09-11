import logging, sys,os
import numpy as np
import psiAnalysisParS2
import p
import set_params
from mpi4py import MPI
COMM = MPI.COMM_WORLD

'''
Copyright (c) Hstau Liao 2019 (python version)    
'''

def split(container, count):
    return [container[j::count] for j in range(count)]

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

def op(proj_name):
    import p
    p.init()
    p.proj_name = proj_name
    set_params.op(1)
    p.create_dir()
    if COMM.rank == 0:
        print("Computing the NLSA snapshots...")
        psiNumsAll = np.tile(np.array(range(p.num_psis)), (p.numberofJobs, 1))  # numberofJobs x num_psis
        sensesAll = np.tile(np.ones(p.num_psis), (p.numberofJobs, 1))  # numberofJobs x num_psis
        rc = {'psiNumsAll': psiNumsAll, 'sensesAll': sensesAll}
        isFull = 0
        fin_PDs = fileCheck(p.numberofJobs)  # array of finished PDs (0's are unfinished, 1's are finished)
        input_data = divid(p.numberofJobs, rc, fin_PDs)
        params1 = dict(isFull=isFull)
    else:
        params1 = None

    params1 = COMM.bcast(params1, root=0)

    if COMM.rank == 0:
        jobs = split(input_data, COMM.size)
    else:
        jobs = None

    jobs = COMM.scatter(jobs, root=0)

    res = []
    for job in jobs:
        isFull = params1['isFull']
        psiAnalysisParS2.op(job, p.conOrderRange, p.trajName, isFull,p.num_psiTrunc)
        res.append([])

    if COMM.rank == 0:
        set_params.op(0)

    MPI.COMM_WORLD.gather(res, root=0)
    return

if __name__ == '__main__':
    op(sys.argv[1])
