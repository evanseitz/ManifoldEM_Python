import manifoldTrimmingAuto
import myio
import numpy as np
import p
import sys, os
import set_params
from mpi4py import MPI
COMM = MPI.COMM_WORLD

'''
Copyright (c) Columbia University Hstau Liao 2019 (python version)
'''

#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

def split(container, count):
    return [container[j::count] for j in range(count)]


def fileCheck():
    fin_PDs = []  # collect list of previously finished PDs from diff_maps/progress/
    for root, dirs, files in os.walk(p.psi_prog):
        for file in sorted(files):
            if not file.startswith('.'):  # ignore hidden files
                fin_PDs.append(int(file))
    return fin_PDs

def divide(N):
    ll=[]
    fin_PDs = fileCheck()
    for prD in range(N):
        dist_file = '{}prD_{}'.format(p.dist_file, prD)
        psi_file = '{}prD_{}'.format(p.psi_file, prD)
        eig_file = '{}/topos/PrD_{}/eig_spec.txt'.format(p.out_dir, prD + 1)
        if prD not in fin_PDs:
            ll.append([dist_file, psi_file, eig_file, prD])
    return ll

def op(proj_name):
    import p
    p.init()
    p.proj_name = proj_name
    set_params.op(1)
    p.create_dir()
    if COMM.rank == 0:
        print("Computing the eigenfunctions...")
        doSave = dict(outputFile='', Is=True)
        # INPUT Parameters
        visual = False
        posPath = 0
        # Finding and trimming manifold from particles
        params1 = dict(doSave=doSave, visual=visual, posPath=posPath)
        input_data = divide(p.numberofJobs)
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
        doSave = params1['doSave']
        visual = params1['visual']
        posPath = params1['posPath']
        manifoldTrimmingAuto.op(job, posPath, p.tune, p.rad, visual, doSave)
        res.append([])

    if COMM.rank == 0:
        set_params.op(0)
    MPI.COMM_WORLD.gather(res, root=0)

    return

if __name__ == '__main__':
    op(sys.argv[1])
