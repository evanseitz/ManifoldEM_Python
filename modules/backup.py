import logging, sys
import myio
import DMembeddingII
import numpy as np
from subprocess import call


#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

'''
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''

def op(PrD,choice):
    import p
    data = myio.fin1(p.tess_file)
    CG = data['CG']
    for j in range(p.num_psis):
        # dirs with frames
        subdir = p.out_dir+'/topos/PrD_{}/psi_{}'.format(PrD,j+1)
        subdir1 = p.out_dir + '/topos/PrD_{}/psi_orig{}'.format(PrD,j+1)
        # movies
        mov_file = p.out_dir + '/topos/PrD_{}/psi_{}.gif'.format(PrD, j + 1)
        mov_orig_file = p.out_dir+'/topos/PrD_{}/psi_{}_orig.gif'.format(PrD,j+1)
        # topos
        tm_file = '{}/topos/PrD_{}/topos_{}.png'.format(p.out_dir, PrD, j + 1)
        tm_orig_file = '{}/topos/PrD_{}/topos_orig_{}.png'.format(p.out_dir, PrD, j + 1)
        if choice == 1:  # need to make backup copy
            # dirs with frames
            call(["rm", "-rf", subdir1])
            call(["cp", "-r", subdir, subdir1])
            # movies
            call(["cp", mov_file, mov_orig_file])
            # topos
            call(["cp", tm_file, tm_orig_file])
            #shutil.rmtree(subdir1)
            #os.renames(subdir, subdir1)
        else:  # restore original backup copy
            # dirs with frames
            call(["rm", "-rf", subdir])
            call(["cp", "-r", subdir1, subdir])
            call(["rm", "-rf", subdir1])
            # movies
            call(["cp", mov_orig_file, mov_file])
            call(["rm", mov_orig_file])
            # topos
            call(["cp", tm_orig_file, tm_file])
            call(["rm", tm_orig_file])

    # diff maps
    psi_file = '{}prD_{}'.format(p.psi_file, PrD-1)
    psi_orig_file = '{}orig_prD_{}'.format(p.psi_file, PrD- 1)
    # psianalysis
    psi2_file = '{}prD_{}'.format(p.psi2_file, PrD - 1)
    psi2_orig_file = '{}orig_prD_{}'.format(p.psi2_file, PrD - 1)
    # class avg
    ca_file = '{}/topos/PrD_{}/class_avg.png'.format(p.out_dir, PrD)
    ca_orig_file = '{}/topos/PrD_{}/class_avg_orig.png'.format(p.out_dir, PrD)
    # eig spectrum
    eig_file = '{}/topos/PrD_{}/eig_spec.txt'.format(p.out_dir, PrD)
    eig_orig_file = '{}/topos/PrD_{}/eig_spec_orig.txt'.format(p.out_dir, PrD)


    if choice == 1:
        call(["cp", psi_file, psi_orig_file])  # create backup copy
        call(["cp", ca_file, ca_orig_file])
        call(["cp", eig_file, eig_orig_file])
        #shutil.copyfile(psi_file, psi_orig_file)
        for i in range(p.num_psis):
            psi2 = '{}_psi_{}'.format(psi2_file, i)
            psi2_orig = '{}_psi_{}'.format(psi2_orig_file, i)
            call(["cp", psi2, psi2_orig])  # create backup copy
    else:
        call(["cp", psi_orig_file, psi_file])  # restore original copy
        call(["rm", psi_orig_file])
        call(["cp", ca_orig_file, ca_file])
        call(["rm", ca_orig_file])
        call(["cp", eig_orig_file, eig_file])
        call(["rm", eig_orig_file])
        for i in range(p.num_psis):
            psi2 = '{}_psi_{}'.format(psi2_file, i)
            psi2_orig = '{}_psi_{}'.format(psi2_orig_file, i)
            call(["cp", psi2_orig, psi2])
            call(["rm", psi2_orig])

