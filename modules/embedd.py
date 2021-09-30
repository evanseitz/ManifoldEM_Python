import logging, sys, os
import myio
import DMembeddingII
import numpy as np
from subprocess import call
import gc
#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

'''
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2020 (python version) 
'''


def op(orig_zip,new_zip,PrD):
    print('Initiating re-embedding...')
    import p
    dist_file = '{}prD_{}'.format(p.dist_file, PrD)
    psi_file = '{}prD_{}'.format(p.psi_file, PrD)
    eig_file = '{}/topos/PrD_{}/eig_spec.txt'.format(p.out_dir, PrD + 1)
    data = myio.fin1(dist_file)
    D = data['D']
    data = myio.fin1(psi_file)
    posPath = data['posPath']
    ind = data['ind']
    D = D[posPath][:,posPath] # D now contains the orig distances
    
    #posPathInd = np.nonzero([x in orig for x in new])[0] # indexes of the new points; this is wrong for Python3 -- E.Seitz, 2021
    
    # Py3 update -- E.Seitz, 2021:
    origX, origY = zip(*orig_zip) #unpack points
    newX, newY = zip(*new_zip) #unpack points
    orig = np.stack((origX, origY), axis=1)
    new = np.stack((newX, newY), axis=1)
    c = np.in1d(orig.view('i,i').reshape(-1), new.view('i,i').reshape(-1))
    cR = np.reshape(c, (int(np.shape(c)[0]/2), 2))
    posPathInd = np.where(cR[:,0])[0] #the ordered indices of 2D-coordinates contained in both lists
    # ...end of Py3 update.     
        
    D1 = D[posPathInd][:,posPathInd] # distances of the new points only
    k = D1.shape[0]
    lamb, psi, sigma, mu, logEps, logSumWij, popt, R_squared = DMembeddingII.op(D1,k,p.tune,60000) #updated 9/11/21
    #print 'old', len(posPath)
    posPath = posPath[posPathInd] # update posPath
    #print 'new',len(posPath)
    lamb = lamb[lamb > 0]

    call(["rm", "-f", eig_file])
    #os.remove(eig_file)
    for i in range(len(lamb) - 1):
        with open(eig_file, "a") as file: #updated 9/11/21
            file.write("%d\t%.5f\n" % (i + 1, lamb[i + 1]))

    myio.fout1(psi_file,['lamb','psi','sigma','mu','posPath','ind'],
                   [lamb,psi,sigma,mu,posPath,ind])

    # remove the existing NLSA and movies etc, so that new ones can be created
    for psinum in range(p.num_psis):
        psi2_file = '{}prD_{}_psi_{}'.format(p.psi2_file, PrD, psinum)
        progress_fname = os.path.join(p.psi2_prog, '%s_%s' % (PrD, psinum))
        call(["rm", "-rf", psi2_file])
        call(["rm", "-rf", progress_fname])
    # class avg
    ca_file = '{}/topos/PrD_{}/class_avg.png'.format(p.out_dir, PrD + 1)
    #file = '{}/topos'.format(p.out_dir)
    #call(['ls','-R', file])
    call(["rm", '-f',ca_file])
    #call(['ls','-R', file])
    #gc.collect()
