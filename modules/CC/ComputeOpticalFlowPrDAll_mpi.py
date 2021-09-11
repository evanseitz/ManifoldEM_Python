import time,os,sys
sys.path.append('modules/')
sys.path.append('modules/CC/')
from ComputeOpticalFlowPrDAll import ComputeOptFlowPrDPsiAll1
from mpi4py import MPI
COMM = MPI.COMM_WORLD
import myio
import set_params
import p

'''
Copyright (c) Columbia University Hstau Liao 2019 (python version)    
'''


def split(container, count):
    return [container[j::count] for j in range(count)]

def divide(N):
    ll = []
    for prD in range(N):
        CC_OF_file = '{}{}'.format(p.CC_OF_file, prD)
        if os.path.exists(CC_OF_file):
            data = myio.fin1(CC_OF_file)
            if data is not None:
                continue
        ll.append([CC_OF_file,prD])
    return ll


def op(proj_name):
    import p
    p.init()
    p.proj_name = proj_name
    set_params.op(1)
    p.create_dir()
    if COMM.rank == 0:
        print("Computing Optical Flow")
        data1 = myio.fin1(p.tess_file)
        CG = data1['CG']
        numberofJobs = len(CG)
        input_data = divide(numberofJobs)
    if COMM.rank == 0:
        jobs = split(input_data, COMM.size)
    else:
        jobs = None
    jobs = COMM.scatter(jobs, root=0)

    res = []
    for job in jobs:
        ComputeOptFlowPrDPsiAll1(job)
        res.append([])
    if COMM.rank == 0:
        set_params.op(0)
    MPI.COMM_WORLD.gather(res, root=0)
    return

if __name__ == '__main__':
    op(sys.argv[1])
