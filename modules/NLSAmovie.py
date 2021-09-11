import logging, sys
import multiprocessing
from subprocess import Popen
import myio
import makeMovie
import matplotlib.pyplot as plt
import numpy as np
import myio
import p
from functools import partial
from contextlib import contextmanager
import gc
import set_params
import time,os
from pyface.qt import QtGui, QtCore
os.environ['ETS_TOOLKIT'] = 'qt4'

'''
% scriptPsiNLSAmovie
% Matlab Version V1.2
% Copyright(c) UWM, Ali Dashti 2016
% This script makes the NLSA movies along each reaction coordinate.
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Sonya Hanson 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''

#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    pool.close()

def divide(N):
    import p
    ll=[]
    for prD in range(N):#prDs
        image_file = '{}/topos/PrD_{}/class_avg.png'.format(p.out_dir, prD + 1)
        if os.path.exists(image_file):
            continue
            #data = myio.fin1(image_file)
            #if data is not None:
            #    continue
        ll.append([prD])
    return ll

def count(N):
    c = 0
    for prD in range(N):
        image_file = '{}/topos/PrD_{}/class_avg.png'.format(p.out_dir, prD + 1)
        if os.path.exists(image_file):
            continue
            #data = myio.fin1(image_file)
            #if data is not None:
            #    continue
        c += 1
    return c

def movie(input_data,out_dir,dist_file,psi2_file,fps):
    prD = input_data[0]
    dist_file1 = '{}prD_{}'.format(dist_file, prD)
    # Fetching NLSA outputs and making movies
    IMG1All = []
    Topo_mean = []
    for psinum in range(p.num_psis):
        psi_file1 = psi2_file + 'prD_{}'.format(prD) + '_psi_{}'.format(psinum)
        data = myio.fin1(psi_file1)
        IMG1All.append(data['IMG1'])
        Topo_mean.append(data['Topo_mean'])
        # make movie
        #outFile = p.movie2d_file +'prD_{}_psi_{}.mp4'.format(p.movie2d_dir, prD, psinum)
        #makeMovie.op(IMG1All[psinum], outFile, p.fps)
        makeMovie.op(IMG1All[psinum], prD, psinum, fps)

        ######################
        # write topos
        topo = Topo_mean[psinum]
        dim = int(np.sqrt(topo.shape[0]))

        fig2 = plt.figure(frameon=False)
        ax2 = fig2.add_axes([0, 0, 1, 1])
        ax2.axis('off')
        ax2.set_title('')
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.imshow(topo[:, 1].reshape(dim, dim), cmap=plt.get_cmap('gray'))
        image_file = '{}/topos/PrD_{}/topos_{}.png'.format(p.out_dir, prD + 1, psinum + 1)
        fig2.savefig(image_file, bbox_inches='tight', dpi=100, pad_inches=-0.1)
        ax2.clear()
        fig2.clf()
        plt.close(fig2)
        #gc.collect()
    # write class avg image
    data = myio.fin1(dist_file1)
    avg = data['imgAvg']
    fig3 = plt.figure(frameon=False)
    ax3 = fig3.add_axes([0, 0, 1, 1])
    ax3.axis('off')
    ax3.set_title('')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    #if p.relion_data == True:
    #    avg = avg.T
    ax3.imshow(avg, cmap=plt.get_cmap('gray'))
    image_file = '{}/topos/PrD_{}/class_avg.png'.format(p.out_dir, prD + 1)
    fig3.savefig(image_file, bbox_inches='tight', dpi=100, pad_inches=-0.1)
    ax3.clear()
    fig3.clf()
    plt.close(fig3)

    return

def op(*argv):
    time.sleep(5)
    import p
    set_params.op(1)
    
    multiprocessing.set_start_method('fork', force=True)

    for i in range(p.numberofJobs):
        for j in range(p.num_psis):
            subdir = p.out_dir + '/topos/PrD_{}/psi_{}'.format(i + 1, j + 1)
            Popen(["mkdir", "-p", subdir])
    if p.machinefile:
        print('using MPI')
        Popen(["mpirun", "-n", str(p.ncpu), "-machinefile", str(p.machinefile),
            "python", "modules/NLSAmovie_mpi.py",str(p.proj_name)],close_fds=True)
        if argv:
            progress4 = argv[0]
            offset = 0
            while offset < p.numberofJobs:
                offset = p.numberofJobs - count(p.numberofJobs)
                progress4.emit(int((offset / float(p.numberofJobs)) * 100))
                time.sleep(5)
    else:
        print("Making the 2D movies...")
        input_data = divide(p.numberofJobs)
        if argv:
            progress4 = argv[0]
            offset = p.numberofJobs - len(input_data)
            progress4.emit(int((offset / float(p.numberofJobs)) * 100))
        if p.ncpu == 1:  # avoids the multiprocessing package
            for i in range(len(input_data)):
                movie(input_data[i],p.out_dir, p.dist_file, p.psi2_file, p.fps)
                if argv:
                    offset += 1
                    progress4.emit(int((offset / float(p.numberofJobs)) * 100))
        else:
            with poolcontext(processes=p.ncpu, maxtasksperchild=1) as pool:
                for i, _ in enumerate(pool.imap_unordered(partial(movie,out_dir=p.out_dir,dist_file=p.dist_file,
                                       psi2_file=p.psi2_file,fps=p.fps), input_data), 1):
                    if argv:
                        offset += 1
                        progress4.emit(int((offset / float(p.numberofJobs)) * 100))
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
