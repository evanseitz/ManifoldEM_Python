import p
import numpy as np
import getDistanceCTF_local_Conj9combinedS2
import os,sys
import myio
import set_params
from mpi4py import MPI
COMM = MPI.COMM_WORLD
import sys
'''
Copyright (c) Columbia University Hstau Liao 2019 (python version)    
'''

def split(container, count):
    return [container[j::count] for j in range(count)]

def fileCheck():
    fin_PDs = [] #collect list of previously finished PDs from distances/progress/
    for root, dirs, files in os.walk(p.dist_prog):
        for file in sorted(files):
            if not file.startswith('.'): #ignore hidden files
                fin_PDs.append(int(file))
    return fin_PDs

def divide(CG,q,df,N):
    ll = []
    fin_PDs = fileCheck()
    for prD in range(N):
        ind = CG[prD]
        q1 = q[:, ind]
        df1 = df[ind]
        dist_file = '{}prD_{}'.format(p.dist_file, prD)
        if prD not in fin_PDs:
            ll.append([ind, q1, df1, dist_file, prD])
    return ll

def op(proj_name):
    import p
    p.init()
    p.proj_name = proj_name
    set_params.op(1)
    p.create_dir()
    if COMM.rank == 0:
        print("Computing the distances...")
        data1 = myio.fin1(p.tess_file)
        CG = data1['CG']
        df = data1['df']
        q = data1['q']
        sh = data1['sh']
        size = len(df)
        filterPar = dict(type='Butter',Qc=0.5,N=8)
        options = dict(verbose=False,avgOnly=False,visual=False,parallel=False,relion_data=p.relion_data,thres=p.PDsizeThH)
        params1 = dict(filterPar=filterPar,options=options,sh=sh,size=size)
        jobs = divide(CG, q, df, p.numberofJobs)
    else:
        params1 = None

    params1 = COMM.bcast(params1, root=0)

    if COMM.rank == 0:
        jobs = split(jobs, COMM.size)
    else:
        jobs = None

    jobs = COMM.scatter(jobs, root=0)

    res = []
    #count = 0
    for job in jobs:
        #print "rank={}".format(COMM.rank)
        filterPar = params1['filterPar']
        sh = params1['sh']
        options = params1['options']
        size = params1['size']
        getDistanceCTF_local_Conj9combinedS2.op(job, filterPar, p.img_stack_file, sh, size, options)
        #count += 1
        #print 'count=',count
        #cc = np.array([count])
        #COMM.Isend(cc, dest=0)
        #if COMM.rank == 0:
            #if argv:
                #print 'offset=',offset
                #for i  in range(COMM.size):
                #   req = COMM.Irecv(cc,source = i)
                #   if req.Test():
                #       print 'type count=',type(cc)
                #       offset += cc[0]
                #print ' o                                                                                                                      ffset=',offset
                ##offset = numberofJobs - count(numberofJobs)
                ##progress1.emit(int((offset / float(numberofJobs)) * 100))
        res.append([])
    if COMM.rank == 0:
        set_params.op(0)
    MPI.COMM_WORLD.gather(res, root=0)

    return

if __name__ == '__main__':
    op(sys.argv[1])
    #op()