import numpy as np
import logging, sys
sys.path.append('../')
import p
import multiprocessing
from functools import partial
from contextlib import contextmanager
import myio
import os
import time

import set_params
from subprocess import Popen, call
import copy
from OpticalFlowMovie import SelectFlowVec
from numpy import linalg as LA
import matplotlib.pyplot as plt
from hogHistogram import histogram_from_gradients, magnitude_orientation,visualise_histogram

#from pyface.qt import QtGui, QtCore
#os.environ['ETS_TOOLKIT'] = 'qt4'


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

@contextmanager

def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


'''
% def CompareOrientMatrix(FlowVecSelA,FlowVecSelB):
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Dec 2017. Modified:Aug 22,2019
  Copyright (c) Columbia University Suvrajit Maji 2020 (python version)
'''

def HOGOpticalFlowPy(flowVec,hogFigfile):

    cellSize = (4,4)  # this is actually the block size
    cellsPerBlock= (2,2)
    visualize = False
    nbins = 9
    signedOrientation = True
    histogramNormalize = True
    flatten=False
    sameSize=True

    hog_params=dict(cell_size=cellSize, cells_per_block=cellsPerBlock, visualise=visualize,nbins=nbins,
                    signed_orientation=signedOrientation, normalise=histogramNormalize,flatten=flatten, same_size=sameSize)

    VxDim = flowVec['Vx'].shape
    #print 'VxDim',VxDim
    if len(VxDim) > 2:
        VxStackDim = VxDim[2]

        tempH =[]
        for d in range(0,VxStackDim):
            gx=flowVec['Vx'][:,:,d]
            gy=flowVec['Vy'][:,:,d]

            tH = histogram_from_gradients(gx, gy,cell_size=cellSize, cells_per_block=cellsPerBlock, visualise=visualize,
                                             nbins=nbins, signed_orientation=signedOrientation, normalise=histogramNormalize,
                                             flatten=flatten, same_size=sameSize)
            tempH.append(tH)


        H = np.array(tempH)
        dims = np.shape(H)
        if len(dims) > 3:
            H = np.moveaxis(H, 0, -1)
    else:

        gx=flowVec['Vx']
        gy=flowVec['Vy']
        H = histogram_from_gradients(gx, gy,cell_size=cellSize, cells_per_block=cellsPerBlock, visualise=visualize,
                                             nbins=nbins, signed_orientation=signedOrientation, normalise=histogramNormalize,
                                             flatten=flatten, same_size=sameSize)


    ## this is very slow it print figure,only for testing purpose
    showHOG = p.opt_movie['OFvisual']
    printHOG = p.opt_movie['printFig']
    if showHOG or printHOG:
        dims = np.shape(H)
        if len(dims)>3:
            nhogFigs = dims[3]
        else:
            nhogFigs=1

        for nh in range(0,nhogFigs):
            if nhogFigs>1:
                hogFigfile_figs = hogFigfile+'_block'+str(nh)
                Hblock = H[:,:,:,nh]
            else:
                hogFigfile_figs = hogFigfile
                Hblock = H
            fig=plt.figure('HOGFeature',figsize=(10,10))
            fig.clf()
            imh2 = visualise_histogram(Hblock, cellSize[0], cellSize[1],signedOrientation)
            plt.title('HOG features')
            plt.imshow(imh2, cmap=plt.cm.Greys_r)
            if showHOG:
                plt.show()
            if printHOG:
                fig.savefig(hogFigfile_figs + '.png')

    return H,hog_params


# plot
def figurePlot(mat):
    fig = plt.figure()
    plt.imshow(mat,cmap=plt.cm.hot)
    plt.show()

# Compare how similar two Matrices/Images are.
# TO DO: Implement error checking for wrong or, improper inputs
# Check for NaN or Inf outputs , etc.
def CompareOrientMatrix(FlowVecSelA,FlowVecSelB,prds_psinums,labels):

    #FlowVxDim = FlowVecSelA['Vx'].shape
    #FlowVxDim = FlowVecSelB['Vx'].shape
    #print 'FlowVxDim',FlowVxDim
    useNorm='l2'
    #if len(FlowVxDim) = 3, then HOGF will be 4-dimensional
    prD_A=prds_psinums[0]
    psinum_A=prds_psinums[1]
    prD_B=prds_psinums[2]
    psinum_B=prds_psinums[3]


    Hog_fig_dir = os.path.join(p.CC_dir,'CC_OF_fig/hog')
    if p.opt_movie['printFig']:
        call(["mkdir", "-p", Hog_fig_dir])

    filenameA = "hog_prd_" + str(prD_A) + '_psi_' + str(psinum_A) + '_' + str(labels[0])
    hogFigfile_A = os.path.join(Hog_fig_dir,filenameA)

    filenameB = "hog_prd_" + str(prD_B) + '_psi_' + str(psinum_B) + '_' + str(labels[1])
    hogFigfile_B = os.path.join(Hog_fig_dir,filenameB)

    HOGFA,hog_params = HOGOpticalFlowPy(FlowVecSelA,hogFigfile_A)
    HOGFB,hog_params = HOGOpticalFlowPy(FlowVecSelB,hogFigfile_B)


    # The dimensions of HOGFA and HOGFB should always match given the number of movie blocks created for movie A and B
    # if for some reason the number of blocks for movie A and B are different, then this check is a fail safe to make
    # the code still work
    hogDimA = HOGFA.shape
    hogDimB = HOGFB.shape
    #print 'hog shape A,B',HOGFA.shape,HOGFB.shape


    #print 'hog shape after adjustment',HOGFA.shape,HOGFB.shape
    hoffset = 1.25
    distHOGAB = []
    distHOGAB_tblock=[]
    isBadPsiAB_block=[]
    hp = np.ceil(np.float(hogDimA[0])/hog_params['cell_size'][0]).astype(int)
    #print 'hp',hp
    num_hogel_th = np.ceil(0.2*(hp**2)*hogDimA[2]).astype(int)
    #print 'num_hogel_th',num_hogel_th

    if useNorm=='l1':
    #L1-norm
        #time block
        if len(hogDimA) > 3:

            distHOGAB_tblock= np.zeros((hogDimA[3],1))
            isBadPsiA_block = np.zeros((hogDimA[3],1))
            isBadPsiB_block = np.zeros((hogDimA[3],1))


            for j in range(0,hogDimA[3]):
                #if  np.count_nonzero(HOGFA[:,:,:,j])==0:
                if  np.count_nonzero(HOGFA[:,:,:,j]) <= num_hogel_th:
                    #print 'HOGFA',j,HOGFA[:,:,:,j]
                    HOGFA[:,:,:,j]= np.random.random(np.shape(HOGFB[:,:,:,j]))+hoffset
                    isBadPsiA_block[j]=1

                #if  np.count_nonzero(HOGFB[:,:,:,j])==0:
                if  np.count_nonzero(HOGFB[:,:,:,j]) <= num_hogel_th:
                    #print 'HOGFB',j,HOGFB[:,:,:,j]
                    HOGFB[:,:,:,j]= np.random.random(np.shape(HOGFB[:,:,:,j]))+hoffset
                    isBadPsiB_block[j]=1

                distHOGAB_tblock[j] = sum(abs(HOGFA[:,:,:,j]-HOGFB[:,:,:,j]))


            isBadPsiAB_block = [isBadPsiA_block.T,isBadPsiB_block.T]

        # this should be done after the adjustments of the zero matrix to a matrix with high random numbers
        distHOGAB = sum(abs(HOGFA-HOGFB))

    if useNorm=='l2':
    #L2-norm

        #time block
        if len(hogDimA) > 3:
            distHOGAB_tblock = np.zeros((hogDimA[3],1))
            isBadPsiA_block = np.zeros((hogDimA[3],1))
            isBadPsiB_block = np.zeros((hogDimA[3],1))
            for j in range(0,hogDimA[3]):

                # hog feature matrix difference for A,B HOGFA - HOGFB will be smaller if either of the two matrices are
                # all zeros, so to produce a maximum difference between a normal feature matrix and such zero
                # feature matrix we can add some random numbers with a high value
                #if  np.count_nonzero(HOGFA[:,:,:,j])==0: # have to check this criteria
                #print 'nonZero_hogA_elements:', np.count_nonzero(HOGFA[:,:,:,j])
                if  np.count_nonzero(HOGFA[:,:,:,j]) <= num_hogel_th:# have to check this criteria
                    #print 'HOGFA',j,HOGFA[:,:,:,j]
                    HOGFA[:,:,:,j]= np.random.random(np.shape(HOGFB[:,:,:,j]))+hoffset
                    isBadPsiA_block[j]=1

                #if  np.count_nonzero(HOGFB[:,:,:,j])==0:
                #print 'nonZero_hogB_elements:',np.count_nonzero(HOGFB[:,:,:,j])
                if  np.count_nonzero(HOGFB[:,:,:,j]) <= num_hogel_th:
                    #print 'HOGFB',j,HOGFB[:,:,:,j]
                    HOGFB[:,:,:,j]= np.random.random(np.shape(HOGFB[:,:,:,j]))+hoffset
                    isBadPsiB_block[j]=1


                distHOGAB_tblock[j] = LA.norm(HOGFA[:,:,:,j]-HOGFB[:,:,:,j])

            isBadPsiAB_block = [isBadPsiA_block.T,isBadPsiB_block.T]

        #print 'isBadPsiAB_block',isBadPsiAB_block
        # this should be done after the adjustments of the zero matrix to a matrix with high random numbers
        distHOGAB = LA.norm(HOGFA-HOGFB)


    #print 'distHOGAB_tblock',distHOGAB_tblock,
    varargout = [distHOGAB,distHOGAB_tblock,isBadPsiAB_block]
    return varargout



def ComparePsiMoviesOpticalFlow(FlowVecSelA,FlowVecSelB,prds_psinums,labels):
    # Analysis of the flow matrix
    psiMovFlowOrientMeasures=dict(Values=[],Values_tblock=[])
    Values,Values_tblock,isBadPsiAB_block = CompareOrientMatrix(FlowVecSelA,FlowVecSelB,prds_psinums,labels)
    psiMovFlowOrientMeasures.update(Values=Values,Values_tblock=Values_tblock)

    return psiMovFlowOrientMeasures,isBadPsiAB_block


def ComputeMeasuresPsiMoviesOpticalFlow(FlowVecSelAFWD,FlowVecSelBFWD,FlowVecSelBREV,prds_psinums):

    labels = ['AFWD','BFWD']
    psiMovOFMeasuresFWD,isBadPsiAB_blockF = ComparePsiMoviesOpticalFlow(FlowVecSelAFWD,FlowVecSelBFWD,prds_psinums,labels)
    psiMovMFWD = psiMovOFMeasuresFWD['Values']
    psiMovMFWD_tblock = psiMovOFMeasuresFWD['Values_tblock']

    labels = ['AFWD','BREV']
    psiMovOFMeasuresREV,isBadPsiAB_blockR = ComparePsiMoviesOpticalFlow(FlowVecSelAFWD,FlowVecSelBREV,prds_psinums,labels)
    psiMovMREV = psiMovOFMeasuresREV['Values']
    psiMovMREV_tblock = psiMovOFMeasuresREV['Values_tblock']

    #print 'prds_psinums',prds_psinums,'isBadPsiAB_blockF',isBadPsiAB_blockF,'isBadPsiAB_blockR',isBadPsiAB_blockR

    psiMovieOFmeasures = dict(MeasABFWD=psiMovMFWD,MeasABFWD_tblock=psiMovMFWD_tblock,MeasABREV=psiMovMREV,MeasABREV_tblock=psiMovMREV_tblock)
    return psiMovieOFmeasures,isBadPsiAB_blockF


def ComputeEdgeMeasurePairWisePsiAll(input_data,G,flowVecPctThresh):

    currPrD = input_data[0]
    nbrPrD = input_data[1]
    CC_meas_file = input_data[2]
    edgeNum = input_data[3]

    currentPrDPsiFile = '{}{}'.format(p.CC_OF_file, currPrD)
    nbrPrDPsiFile = '{}{}'.format(p.CC_OF_file, nbrPrD)

    NumPsis = p.num_psis
    #print 'NumPsis',NumPsis
    #load the data for the current and neighbor prds
    #print '\nLoading data for current and neighbor PrD, Edge:{} ...'.format(edgeNum)
    data = myio.fin1(currentPrDPsiFile)
    FlowVecCurrPrD = data['FlowVecPrD']
    data = myio.fin1(nbrPrDPsiFile)
    FlowVecNbrPrD = data['FlowVecPrD']

    nEdges = G['nEdges']

    if len(FlowVecCurrPrD[0]['FWD']['Vx'].shape)>2:
        numtblocks=FlowVecCurrPrD[0]['FWD']['Vx'].shape[2]
    else:
        numtblocks=1

    measureOFCurrNbrFWD = np.empty((nEdges,NumPsis,NumPsis))
    measureOFCurrNbrREV = np.empty((nEdges,NumPsis,NumPsis))

    if numtblocks > 1:
        #numtblocks = 3 # temp , have to fix this from beginning
        measureOFCurrNbrFWD_tblock = np.empty((nEdges,NumPsis,NumPsis*numtblocks))
        measureOFCurrNbrREV_tblock = np.empty((nEdges,NumPsis,NumPsis*numtblocks))

    psiSelcurrPrD = range(NumPsis)
    psiCandidatesNnbrPrD = range(NumPsis) # in case psis for currPrD is different from nbrPrD


    badNodesPsisBlock = np.zeros((G['nNodes'],NumPsis))

    #loop over for psi selections for current prD
    for psinum_currPrD in psiSelcurrPrD:

        if FlowVecCurrPrD[psinum_currPrD]['FWD']: # check if this condition holds for all kind of entries of the dict
            FlowVecCurrPrDFWD = SelectFlowVec(FlowVecCurrPrD[psinum_currPrD]['FWD'],flowVecPctThresh)

        # psi selection candidates for the neighboring prD
        for psinum_nbrPrD in psiCandidatesNnbrPrD:
            #print('\nCurrent-PrD:{}, Current-PrD-Psi:{}\nNeighbor-PrD:{}, Neighbor-PrD-Psi:{}'.format(currPrD, psinum_currPrD, nbrPrD, psinum_nbrPrD))
            if FlowVecNbrPrD[psinum_nbrPrD]['REV']:
                FlowVecNbrPrDFWD = SelectFlowVec(FlowVecNbrPrD[psinum_nbrPrD]['FWD'],flowVecPctThresh)
                FlowVecNbrPrDREV = SelectFlowVec(FlowVecNbrPrD[psinum_nbrPrD]['REV'],flowVecPctThresh)

            prds_psinums = [currPrD, psinum_currPrD, nbrPrD, psinum_nbrPrD]

            FlowVecSelAFWD = FlowVecCurrPrDFWD
            FlowVecSelBFWD = FlowVecNbrPrDFWD
            FlowVecSelBREV = FlowVecNbrPrDREV

            psiMovieOFmeasures,isBadPsiAB_block = ComputeMeasuresPsiMoviesOpticalFlow(FlowVecSelAFWD,FlowVecSelBFWD,FlowVecSelBREV,prds_psinums)
            #print('psiMovieOFmeasures:',psiMovieOFmeasures,'\n')
            measureOFCurrNbrFWD[edgeNum][psinum_currPrD,psinum_nbrPrD] = psiMovieOFmeasures['MeasABFWD']
            measureOFCurrNbrREV[edgeNum][psinum_currPrD,psinum_nbrPrD] = psiMovieOFmeasures['MeasABREV']

            if numtblocks > 1:
                badNodesPsisBlock[currPrD,psinum_currPrD] = -100*np.sum(isBadPsiAB_block[0])
                badNodesPsisBlock[nbrPrD,psinum_nbrPrD] = -100*np.sum(isBadPsiAB_block[1])

            if numtblocks > 1:
                t = psinum_nbrPrD*numtblocks
                #time block
                print('t',t,'numtblocks',numtblocks)
                measureOFCurrNbrFWD_tblock[edgeNum][psinum_currPrD,t:t+numtblocks] = np.transpose(psiMovieOFmeasures['MeasABFWD_tblock'])
                measureOFCurrNbrREV_tblock[edgeNum][psinum_currPrD,t:t+numtblocks] = np.transpose(psiMovieOFmeasures['MeasABREV_tblock'])


    measureOFCurrNbrEdge = np.hstack((measureOFCurrNbrFWD[edgeNum],measureOFCurrNbrREV[edgeNum]))

    if numtblocks > 1:
        measureOFCurrNbrEdge_tblock = np.hstack((measureOFCurrNbrFWD_tblock[edgeNum],measureOFCurrNbrREV_tblock[edgeNum]))
    else:
        measureOFCurrNbrEdge_tblock = []

    #print '\nCurrent-PrD:{}, Neighbor-PrD:{}'.format(currPrD, nbrPrD)
    #print 'measureOFCurrNbrEdge',measureOFCurrNbrEdge
    #print 'measureOFCurrNbrEdge_tblock',measureOFCurrNbrEdge_tblock

    #return
    #print 'Writing edge-{} data to file'.format(edgeNum)
    myio.fout1(CC_meas_file, ['measureOFCurrNbrEdge','measureOFCurrNbrEdge_tblock','badNodesPsisBlock'], [measureOFCurrNbrEdge,measureOFCurrNbrEdge_tblock,badNodesPsisBlock])

    #######################################################
    # create empty PD files after each Pickle dump to...
    # ...be used to resume (avoiding corrupt Pickle files):
    progress_fname = os.path.join(p.CC_meas_prog, '%s' % (edgeNum))
    open(progress_fname, 'a').close() #create empty file to signify non-corrupted Pickle dump
    #######################################################


# changed Nov 30, 2018, S.M.
# here N is a list of (edge) numbers
def divide1(N, G):
    ll = []

    fin_edges = [] #collect list of previously finished PDs from CC/CC_meas/
    for root, dirs, files in os.walk(p.CC_meas_prog):
        for file in sorted(files):
            if not file.startswith('.'): #ignore hidden files
                fin_edges.append(int(file))

    for e in N:
        currPrD = G['Edges'][e, 0]
        nbrPrD = G['Edges'][e, 1]
        CC_meas_file = '{}{}_{}_{}'.format(p.CC_meas_file, e, currPrD, nbrPrD)

        if e in fin_edges:
            continue
        else:
            ll.append([currPrD,nbrPrD,CC_meas_file,e])

    return ll

def count1(R, G):
    c = 0
    for e in R:
        currPrD = G['Edges'][e, 0]
        nbrPrD = G['Edges'][e, 1]
        CC_meas_file = '{}{}_{}_{}'.format(p.CC_meas_file, e, currPrD, nbrPrD)
        if os.path.exists(CC_meas_file):
            data = myio.fin1(CC_meas_file)
            if data is not None:
                continue
        c+=1
    return c

def op(G, nodeEdgeNumRange, *argv):
    time.sleep(1)

    multiprocessing.set_start_method('fork', force=True)

    set_params.op(1)
    #set_params.op(-1)
    import p
    set_params.op(1)

    nodeRange = nodeEdgeNumRange[0]
    edgeNumRange = nodeEdgeNumRange[1]
    if len(edgeNumRange) == 0:
        edgeNumRange = range(G['nEdges'])
    numberofJobs = len(nodeRange) + len(edgeNumRange)

    if p.machinefile:
        print('using MPI')
        Popen(["mpirun", "-n", str(p.ncpu), "-machinefile", str(p.machinefile),
              "python", "modules/CC/ComputeMeasureEdgeAll_mpi.py"],close_fds=True)
        if argv:
            progress5 = argv[0]
            offset = 0
            while offset < numberofJobs:
                offset = numberofJobs - count1(edgeNumRange,G)
                progress5.emit(int((offset / float(numberofJobs)) * 99))
                time.sleep(15)
    else:

        flowVecPctThresh = p.opt_movie['flowVecPctThresh']

        if argv:
            progress5 = argv[0]

        #extract info for psi selection/sense of ref and psi candidates for nbr
        input_data = divide1(edgeNumRange,G) # changed Nov 30, 2018, S.M.


        if argv:
            offset = numberofJobs - len(input_data)
            #print 'edge meas offset',offset
            progress5.emit(int((offset / float(numberofJobs)) * 99))

        if p.ncpu == 1:  # avoids the multiprocessing package
            for i in range(len(input_data)):
                ComputeEdgeMeasurePairWisePsiAll(input_data[i],G,flowVecPctThresh)
                if argv:
                    offset += 1
                    progress5.emit(int((offset / float(numberofJobs)) * 99))
        else:
            with poolcontext(processes=p.ncpu,maxtasksperchild=1) as pool:
                for i, _ in enumerate(pool.imap_unordered(partial(ComputeEdgeMeasurePairWisePsiAll,G=G,
                                           flowVecPctThresh=flowVecPctThresh), input_data),1):
                    if argv:
                        offset += 1
                        progress5.emit(int((offset/ float(numberofJobs)) * 99))
                    time.sleep(0.05)
                pool.close()
                pool.join()

    set_params.op(0)
