import logging, sys
import makeMovie
import matplotlib.pyplot as plt
import numpy as np
import myio
import p
import time,os,sys
import set_params
from mpi4py import MPI
COMM = MPI.COMM_WORLD

'''
Copyright (c) Columbia University Hstau Liao 2019 (python version)
'''

def split(container, count):
    return [container[j::count] for j in range(count)]


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
        ax2.imshow(topo[:, 1].reshape(dim, dim).T, cmap=plt.get_cmap('gray'))
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


def op(proj_name):
    import p
    p.init()
    p.proj_name = proj_name
    set_params.op(1)
    p.create_dir()
    if COMM.rank == 0:
        print("Making the 2D movies...")
        input_data = divide(p.numberofJobs)
        jobs = split(input_data, COMM.size)
    else:
        jobs = None

    jobs = COMM.scatter(jobs, root=0)

    res = []
    for job in jobs:
        movie(job, p.out_dir, p.dist_file, p.psi2_file, p.fps)
        res.append([])
    if COMM.rank == 0:
        set_params.op(0)
    MPI.COMM_WORLD.gather(res, root=0)
    return

if __name__ == '__main__':
    op(sys.argv[1])
