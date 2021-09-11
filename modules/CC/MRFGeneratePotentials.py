import numpy as np
import logging
import p
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

'''
function [nodePot,edgePot] = MRFGeneratePotentials(G,anchorNodes,anchorNodeMeasures,edgeMeasures)
% Generate the node and edge potential for the Markov Random Field graphical model
% Input: 
%   G: Input graph structure
%   anchorNodes:
%   anchorNodeMeasures: measures to be used for node potentials (for known nodes only) 
%   edgeMeasures: measures to be used for edge potentials (e.g. optical flow
%   orientation HoG of the psi-movies)
%   methodNum: which measure type for optical flow orientation computation
%   to be used
% Output: 
%   nodePot: s x n node potential, s = number of states , n is number of nodes  
%   edgePot: s x s x e edge potential,s = number of states , e is number of edges  
%
%
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Feb 02,2018. Modified:Nov 28,2018
Copyright (c) Columbia University Suvrajit Maji 2018 
(original matlab version and python debugging)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    

'''
    #% node potential function, customize it
def nodePotentialFunction(M,beta):
    npotfn = np.exp(np.dot(beta,M))
    return npotfn

def edgePotentialFunction(M,beta):
    epotfn = np.exp(-np.dot(beta,(1-M)))
    return epotfn

    #% edge potential function, customize it
def transformFunction_simple(M):
    kexpt = np.exp(-M) # default kernel
    return kexpt

def transformFunction(M,elist,printPotFig):
    #print('Plotting the transform function of edge measurements')
    import matplotlib.pyplot as plt
    e,n1,n2=elist[:]

    sigma = 1.2
    #print('sigma',sigma)
    Mt = np.exp(-M/(2.0*sigma**2)) #modified kernel
    #print('med-Mt',np.median(Mt))
    nBlocks = np.ceil(M.shape[1]/(2.0*p.num_psis)).astype(int)

    showPlot = 0#printPotFig
    if showPlot or printPotFig:
        #print('Now printing Edge Potential figures...')
        fig= plt.figure('EdgePot',figsize=(10,10))
        fig.clf()
        plt.imshow(Mt,cmap='jet',interpolation='nearest')
        #plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        xticklabels = np.arange(0,M.shape[1],nBlocks)+1  # 1 indexing
        plt.xticks(np.arange(0,M.shape[1],nBlocks),xticklabels)
        #yticklabels = np.arange(0,M.shape[1])+1 # 1 indexing
        #plt.yticks(np.arange(0,p.num_psis,1),yticklabels)
        plt.rcParams.update({'font.size': 18})
        plt.colorbar()
        plt.xlabel('Frame-blocks of {}, PD-{}, psi 1 to {}'.format(nBlocks,n2+1,p.num_psis))  # n2 : 1 indexing
        plt.ylabel('PD-{}, psi'.format(n1+1))  # n1:1 indexing

        if showPlot:
            plt.show()

        if printPotFig:
            import os
            from subprocess import call
            e,n1,n2=elist[:]
            CC_meas_dir = os.path.join(p.CC_dir,'CC_meas_fig/')
            call(["mkdir", "-p", CC_meas_dir])
            potfilename='pot_edge'+str(e+1)+'_'+str(n1+1)+'_'+str(n2+1)
            potfile = os.path.join(CC_meas_dir,potfilename)
            fig.savefig(potfile + '.png')
    return Mt

def transformFunction_tblock(M,elist,label,printPotFig):
    #print('Plotting the transform function of edge measurements')
    import matplotlib.pyplot as plt
    e,n1,n2=elist[:]

    import time
    sigma = 1.25
    #print('sigma',sigma)
    Mt = np.exp(-M/(2.0*sigma**2)) #modified kernel

    nBlocks = np.ceil(M.shape[1]/(2.0*p.num_psis)).astype(int)

    #print('med-Mt',np.median(Mt))
    showPlot=0 #only show for checking
    printPotFig=0 # only print for checking
    if showPlot or printPotFig:
        #print('Now printing Edge Potential figures...')
        fig= plt.figure('EdgePot_'+label,figsize=(20,10))
        fig.clf()
        plt.imshow(Mt,cmap='jet',interpolation='nearest')
        #plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        xticklabels = np.arange(0,M.shape[1],nBlocks)+1  # 1 indexing
        plt.xticks(np.arange(0,M.shape[1],nBlocks),xticklabels)
        yticklabels = np.arange(0,M.shape[1])+1  # 1 indexing
        plt.yticks(np.arange(0,p.num_psis,1),yticklabels)
        plt.rcParams.update({'font.size': 18})
        plt.colorbar()
        plt.xlabel('Frame-blocks of {}, PD-{}, psi 1 to {}'.format(nBlocks,n2+1,p.num_psis))  # n2: 1 indexing
        plt.ylabel('PD-{}, psi'.format(n1+1))  # n1:1 indexing

        if showPlot:
            plt.show()

        if printPotFig:
            import os
            from subprocess import call
            CC_meas_dir = os.path.join(p.CC_dir,'CC_meas_fig/')
            call(["mkdir", "-p", CC_meas_dir])
            potfilename=label+'pot_edge'+str(e+1)+'_'+str(n1+1)+'_'+str(n2+1)
            potfile = os.path.join(CC_meas_dir,potfilename)
            fig.savefig(potfile + '.png')

    return Mt

def op(G,anchorNodes,anchorNodeMeasures,edgeMeasures,edgeMeasures_tblock):
    Edges = G['Edges']
    NumPsis = p.num_psis
    maxState = 2*NumPsis
    nNodes = G['nNodes']
    nEdges = Edges.shape[0]

    #% generate node potentials
    #% uniform 'prior' for all unobserved nodes
    nP = np.zeros((maxState,nNodes))
    if len(anchorNodes)>0:
        for i in range(len(anchorNodes)):
            nP[:,anchorNodes[i]] = anchorNodeMeasures[:,i]

    #% for known nodes use the provided potential
    nodePot = nodePotentialFunction(nP,1)

    #% generate edge potentials
    edgePot = np.zeros((nEdges,maxState,maxState))+ 1e-10  # flip the dimensions for Python conv
    # For the empty edges (fow which we do not have any measures) , we cannot just leave it as zeros as it will cause
    # Nan error because of 'Normalize' function. Se we add 'eps' to zeros

    nodesEmptyMeas =[]
    for e in range(nEdges): # change to edgeNumRange later
        n1 = Edges[e,0]
        n2 = Edges[e,1]
        #print('edge',e,n1,n2)
        if n1 < n2:
            #mOF = edgeMeasures[:,:,e] # for matlab input edgeMeasures{n1,n2}
            if edgeMeasures[e] is not None:
                mOF = np.asarray(edgeMeasures[e])
            else:
                mOF = np.asarray([])
            if edgeMeasures_tblock[e] is not None:
                mOF_tblock = np.asarray(edgeMeasures_tblock[e])
            else:
                mOF_tblock = np.asarray([])

        else:
            # Note: we do not need the if-else condition here like the Matlab code, since we are using a different data
            # structure for edgeMeasure(which depends on just the edge number here)
            #mOF = edgeMeasures[:,:,e] # for matlab input edgeMeasures{n2,n1}
            #% since it is symmetric computation for nodes n1 and n2
            if edgeMeasures[e] is not None:
                mOF = np.asarray(edgeMeasures[e])
            else:
                mOF = np.asarray([])
            if edgeMeasures_tblock[e] is not None:
                mOF_tblock = np.asarray(edgeMeasures_tblock[e])
            else:
                mOF_tblock = np.asarray([])


        #print('edge num:',e,'nodes:',n1,n2)
        if (mOF is not None) and (mOF.size>0):
            mOFn = transformFunction(mOF,[e,n1,n2],p.opt_movie['printFig'])

            if (mOF_tblock is not None) and (mOF_tblock.size>0):
                mOFn_tblock = transformFunction_tblock(mOF_tblock,[e,n1,n2],'tblock',p.opt_movie['printFig'])

            w = [1.0, 0.0]
            measOF = w[0]*mOFn
            Medge = np.vstack((measOF,np.hstack((measOF[:,NumPsis:], measOF[:,:NumPsis]))))
            edgePot[e,:,:] = Medge

    #print('edgePot',edgePot)
    return (nodePot,edgePot)

