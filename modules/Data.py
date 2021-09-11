import logging, sys, os
import util
import numpy as np
import S2tessellation
import read_alignfile
import myio
import datetime
import math
import numpy as np
from subprocess import call
import FindCCGraph
import set_params

'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)   
Copyright (c) Columbia University Evan Seitz 2019 (python version)
Copyright (c) Columbia University Suvrajit Maji 2019 (python version)
'''

#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

def cart2sph(x, y, z):
        r = math.sqrt(x**2 + y**2 + z**2)
        phi = math.atan2(y,x)*180./math.pi
        theta = math.acos(z/r)*180./math.pi   #it was theta
        return (r, phi, theta)

def genColorConnComp(G):
    numConnComp = len(G['NodesConnComp'])

    nodesColor = np.zeros((G['nNodes'],1),dtype='int')
    for i in range(numConnComp):
        nodesCC = G['NodesConnComp'][i]
        #print 'nodesCC',nodesCC
        nodesColor[nodesCC]=i

    return nodesColor

def write_angles(ang_file,color,S20,full,NC):
    call(["rm", "-f", ang_file])

    if full == 1: #already thresholded S20
        L = range(0,S20.shape[1])
    else: #full S20, still need to take the correct half
        mid = np.floor(S20.shape[1] / 2).astype(int)
        NC1 = NC[:int(mid)]
        NC2 = NC[int(mid):]
        if len(NC1) >= len(NC2): #first half of S2
            L = range(0,mid)
        else:
            L = range(mid,int(S20.shape[1])) #second half of S2

    prD_idx = 0 #needs to always start at 0 regardless of which half used above
    for prD in L:
        x = S20[0, prD]
        y = S20[1, prD]
        z = S20[2, prD]
        r, phi, theta = cart2sph(x, y, z)

        if full:
            prDColor = color[prD]
        else:
            prDColor = int(0)
	
        with open(ang_file, "a") as file:
            file.write("%d\t%.2f\t%.2f\t%d\t%.4f\t%.4f\t%.4f\t%d\n" % (prD_idx + 1, theta, phi, int(0), x, y, z, prDColor))
        prD_idx += 1

def op(align_param_file):
    import p
    set_params.op(1)
    visual = False

    if not p.relion_data: # assumes SPIDER data
        # read the angles
        q = read_alignfile.get_q(align_param_file, p.phiCol, p.thetaCol, p.psiCol,flip=True)
        # double the number of data points by augmentation
        q = util.augment(q)
        # read defocus
        df = read_alignfile.get_df(align_param_file, p.dfCol)
        # double the number of data points by augmentation
        df = np.concatenate((df, df))
        sh = read_alignfile.get_shift(align_param_file,p.shx_col,p.shy_col)
        size = len(df)
    else:
        sh,q,U,V = read_alignfile.get_from_relion(align_param_file,flip=True)
        df = (U+V)/2
        # double the number of data points by augmentation
        q = util.augment(q)
        df = np.concatenate((df, df))
        size = len(df)

    CG1, CG, nG, S2, S20_th, S20, NC = S2tessellation.op(q, p.ang_width, p.PDsizeThL, visual, p.PDsizeThH)
    # CG1: list of lists, each of which is a list of image indices within one PD
    # CG: thresholded version of CG1
    # nG: approximate number of tessellated bins
    # S2: cartesian coordinates of each of particles' angular position on S2 sphere
    # S20_th: thresholded version of S20
    # S20: cartesian coordinates of each bin-center on S2 sphere
    # NC: list of occupancies of each PD
    
    # copy ref angles S20 to file
    #print 'S20_shape=',S20_th.shape
    #print 'CG_shape=', len(CG)

    nowTime = datetime.datetime.now()
    nowTime = nowTime.strftime("%d-%b-%Y %H:%M:%S")
    
    p.nowTime_file = os.path.join(p.user_dir,'outputs_{}/nowTime'.format(p.proj_name))
    myio.fout1(p.nowTime_file,['nowTime'],[nowTime])
    #p.tess_file = 'selecGCs_{}'.format(nowTime)
    p.tess_file = os.path.join(p.user_dir,'outputs_{}/selecGCs'.format(p.proj_name))
    
    myio.fout1(p.tess_file,['CG1', 'CG', 'nG', 'q', 'df', 'S2', 'S20', 'sh', 'NC'],
               [CG1, CG, nG, q, df, S2, S20_th, sh, NC])

    #print 'tessfile=',p.tess_file
    p.numberofJobs = len(CG)
    set_params.op(0)

    if p.resProj == 0 and (np.shape(CG)[0] > 2):
        G,Gsub = FindCCGraph.op()
        nodesColor = genColorConnComp(G)

        write_angles(p.ref_ang_file,nodesColor,S20_th,1,NC) #to PrD_map.txt (thresh bins)
        write_angles(p.ref_ang_file1,nodesColor,S20,0,NC) #to PrD_map1.txt (all bins)

if __name__ == '__main__':
    import p, os
    p.init()
    p.user_dir = '../'
    p.out_dir = os.path.join(p.user_dir, 'outputs_{}/'.format(p.proj_name))
    p.tess_file = '{}/selecGCs'.format(p.out_dir)
    p.nowTime_file = os.path.join(p.user_dir, 'outputs_{}/nowTime'.format(p.proj_name))
    p.align_param_file = os.path.join(p.user_dir, 'data_input/Alignments/few2.tls')
    p.create_dir()
    op(p.align_param_file)
