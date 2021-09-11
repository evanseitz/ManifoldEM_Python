import time,os,sys
sys.path.append('modules/')
sys.path.append('modules/RC/')
from ComputeMeasureEdgeAll import ComputeEdgeMeasurePairWisePsiAll
from mpi4py import MPI
COMM = MPI.COMM_WORLD
import p
import myio
import set_params

'''
Copyright (c) Columbia University Hstau Liao 2019 (python version)    
'''

def split(container, count):
    return [container[j::count] for j in range(count)]

def divide1(R, G):
    ll = []
    for e in R:
        currPrD = G['Edges'][e, 0]
        nbrPrD = G['Edges'][e, 1]
        CC_meas_file = '{}{}_{}_{}'.format(p.CC_meas_file, e, currPrD, nbrPrD)
        if os.path.exists(CC_meas_file):
            data = myio.fin1(CC_meas_file)
            if data is not None:
                continue
        ll.append([currPrD,nbrPrD,CC_meas_file,e])
    return ll


def op(proj_name):
    import p
    p.init()
    p.proj_name = proj_name
    set_params.op(1)
    p.create_dir()
    if COMM.rank == 0:
        print("Computing Edge Measurements")
        data1 = myio.fin1(p.CC_graph_file)
        G = data1['G']
        edgeNumRange = data1['edgeNumRange']
        flowVecPctThresh = p.opt_movie['flowVecPctThresh']
        input_data = divide1(edgeNumRange,G)
        params1 = dict(G=G, flowVecPctThresh=flowVecPctThresh)
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
        G = params1['G']
        flowVecPctThresh = params1['flowVecPctThresh']
        ComputeEdgeMeasurePairWisePsiAll(job,G,flowVecPctThresh)
        res.append([])
    if COMM.rank == 0:
        set_params.op(0)
    MPI.COMM_WORLD.gather(res, root=0)
    return

if __name__ == '__main__':
    op(sys.argv[1])
