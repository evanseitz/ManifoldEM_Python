import logging, sys
import numpy as np
import psiAnalysisParS2
import myio
import set_params
import time,os,sys
import p
from mpi4py import MPI
COMM = MPI.COMM_WORLD

'''
Copyright (c) Columbia University Hstau Liao 2019 (python version)    
'''


def split(container, count):
    return [container[j::count] for j in range(count)]

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

def op(proj_name):
    import p
    p.init()
    p.proj_name = proj_name
    set_params.op(1)
    p.create_dir()
    if COMM.rank == 0:
        print("Recomputing the NLSA snapshots using the found reaction coordinates...")
        data = myio.fin1(p.CC_file)
        psiNumsAll = data['psinums']
        sensesAll = data['senses']
        #rc = {'psiNumsAll': psiNumsAll, 'sensesAll': sensesAll}
        isFull = 1
        R = np.array(range(p.numberofJobs))
        R = np.delete(R, np.nonzero(p.trash_list == 1)[0])
        input_data = divide1(R, psiNumsAll, sensesAll)
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
