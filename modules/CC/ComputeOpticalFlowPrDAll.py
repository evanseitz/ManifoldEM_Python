import numpy as np
import logging, sys
import myio
import datetime
import multiprocessing
from functools import partial
from contextlib import contextmanager
from subprocess import Popen, call
import operator
import os
import time
import OpticalFlowMovie
from OpticalFlowMovie import getOrientMag
import LoadPrDPsiMoviesMasked
sys.path.append('../')
import mrcfile
import set_params
import p
import copy

#from pyface.qt import QtGui, QtCore
#os.environ['ETS_TOOLKIT'] = 'qt4'

sys.path.append('../')

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


@contextmanager

def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

'''
# this only works when PrD are in sequence
# here N is a number (total nodes)
def divide(N):
    ll = []
    for prD in range(N):
        #CC_OF_file = '{}/prD_{}'.format(p.CC_OF_dir, prD)
	    CC_OF_file = '{}{}'.format(p.CC_OF_file, prD)
        ll.append([CC_OF_file,prD])
    return ll
'''

# changed Nov 30, 2018, S.M.
# here N is a list of (node) numbers
def divide1(R):
    ll = []
    for prD in R:
        CC_OF_file = '{}{}'.format(p.CC_OF_file, prD)
        if os.path.exists(CC_OF_file):
            data = myio.fin1(CC_OF_file)
            if data is not None:
                continue
        ll.append([CC_OF_file,prD])
    return ll


def count1(R):
    c = 0
    for prD in R:
        CC_OF_file = '{}{}'.format(p.CC_OF_file, prD)
        if os.path.exists(CC_OF_file):
            data = myio.fin1(CC_OF_file)
            if data is not None:
                continue
        c+=1
    return c

'''
function ComputeOpticalFlowPrDAll
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: May 2018. Modified:Aug 16,2019
'''


def stackDicts(a, b, op=operator.concat):
    #op=lambda x,y: np.dstack((x,y),axis=2)
    op=lambda x,y: np.dstack((x,y))
    mergeDict = dict(a.items() + b.items() + [(k, op(a[k], b[k])) for k in set(b) & set(a)])
    return mergeDict


def ComputePsiMovieOpticalFlow(Mov, opt_movie, prds_psinums):

    OFvisualPrint = [opt_movie['OFvisual'], opt_movie['printFig']]
    Labels = ['FWD','REV']

    computeOF = 1
    blockSize_avg = 5 #how many frames will used for normal averaging
    currPrD = prds_psinums[0]
    psinum_currPrD = prds_psinums[1]
    prd_psinum = [currPrD, psinum_currPrD]

    #print '\nprd_psinum',prd_psinum
    MFWD = copy.deepcopy(Mov) #FWD
    #MREV = Mov[::-1,:] #REV
    numFrames = Mov.shape[0]
    #overlapFrames =  np.ceil(0.40*numFrames).astype(int)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Compute the Optical Flow vectors for each movie
    # For complicated motion involving some rotation component, the 2d movie can be misleading if we get the
    # optical flow vector added over the entire movie , so we might split the movie into blocks and compare
    # the vectors separtely , with splitmovie = 1, this is experimental now.

    # at present the stacking of the dictionary for two blocks has been checked, for others needs to be verified
    splitmovie = 0
    FlowVecFWD = []
    FlowVecREV = []
    if computeOF:

        if splitmovie:
            # number of frames in each blocks
            #blockSize_split = 20#25#np.ceil(numFrames/2).astype(int)+1 # except for the last block, remaining blocks will be of this size
            numBlocks_split = 3
            overlapFrames = 0 #12

            blockSize_split = np.round(np.float(numFrames + (numBlocks_split-1)*overlapFrames + 1)/(numBlocks_split)).astype(int)
            #numBlocks_split = np.round(np.float(numFrames - blockSize_split  + 1)/(blockSize_split - overlapFrames)).astype(int) + 1 + 1

            # In case we fix blockSize, it should be noted that the numBlocks will be different for different
            # blocksize and overlap values
            # Also, one extra block is used in case there is 1 or 2 frames left over after frameEnd is close to
            # numFrames and a new block is created with overlapping frames till frameEnd = numFrames
            # TO DO:better handling of this splitting into overlapping blocks
            for b in range(0,numBlocks_split):
                #frameStart = max(0,b*overlapFrames)
                #frameEnd = min(b*overlapFrames + blockSize_split - 1,numFrames)
                frameStart = max(0,b*(blockSize_split-overlapFrames))
                frameEnd = min(b*(blockSize_split-overlapFrames) + blockSize_split - 1,numFrames)

                if numFrames-frameEnd<5:
                    frameEnd = numFrames
                # check this criteria
                if  frameEnd - frameStart > 0:
                    #print 'frameStart,frameEnd', frameStart,frameEnd
                    blockMovieFWD = MFWD[frameStart:frameEnd,:]
                    #blockMovieREV = MREV[frameStart:frameEnd,:]

                    #print('Computing Optical Flow for Movie Forward ...')
                    FlowVecFblock = OpticalFlowMovie.op(blockMovieFWD,prd_psinum,blockSize_avg,Labels[0]+'-H'+ str(b),OFvisualPrint)
                    if b==0:
                        FlowVecFWD = copy.deepcopy(FlowVecFblock)
                    else:
                        FlowVecFWD = stackDicts(FlowVecFWD,FlowVecFblock)

                    #print('Computing Optical Flow for Movie Reverse ...')
                    # blockMovieFWD is used but due to label of 'REV', the negative vectors will be used after computing the FWD vectors
                    # If FWD vectors are provided, then reverse flow vectors are not going to be recomputed but will
                    # be obtained by reversing the FWD vectors (-Vx,-Vy)
                    # use FlowVecFblock as it is just one block, FlowVecFWD for multiple blocks has multidimensional Vx and Vy--stacked
                    FlowVecRblock = OpticalFlowMovie.op(blockMovieFWD,prd_psinum,blockSize_avg,Labels[1]+'-H'+ str(b),OFvisualPrint,FlowVecFblock)

                    if b==0:
                        FlowVecREV = copy.deepcopy(FlowVecRblock)
                    else:
                        FlowVecREV = stackDicts(FlowVecREV,FlowVecRblock)

                if frameEnd==numFrames:
                    break

        else:
            FlowVecFWD = OpticalFlowMovie.op(MFWD,prd_psinum,blockSize_avg,Labels[0],OFvisualPrint)

            #FlowVecREV = OpticalFlowMovie.op(MREV,prd_psinum,blockSize_avg,Labels[1],OFvisualPrint)
            # MFWD is used but due to label of 'REV', the negative vectors will be used after getting the FWD vectors
            FlowVecREV = OpticalFlowMovie.op(MFWD,prd_psinum,blockSize_avg,Labels[1],OFvisualPrint,FlowVecFWD)


    else:
        print('')
        #print 'Using the previously computed Optical Flow vectors for Movie A ...'
        #print 'Using the previously computed Optical Flow vectors for Movie B ...'

    #print 'FlowVecFWD.shape',np.shape(FlowVecFWD['Vx']),FlowVecFWD['Vx']
    FlowVec = dict(FWD=FlowVecFWD,REV=FlowVecREV)

    return FlowVec


def ComputeOptFlowPrDPsiAll1(input_data):
    time.sleep(5)
    CC_OF_file = input_data[0]
    currPrD = input_data[1]
    FlowVecPrD = np.empty(p.num_psis,dtype=object)
    psiSelcurrPrD = range(p.num_psis)

    #print ('currPrD',currPrD)
    #load movie and tau param first
    moviePrDPsi, badPsis, tauPrDPsis, tauPsisIQR, tauPsisOcc  = LoadPrDPsiMoviesMasked.op(currPrD)

    #print 'curr PD',currPrD
    badPsis = np.array(badPsis)
    #print('badPsis',badPsis,len(badPsis),tauPsisIQR)
    #print('badPsis for prD',currPrD,badPsis,len(badPsis),tauPsisIQR)
    CC_dir_temp = '{}temp/'.format(p.CC_dir)
    #print(CC_dir_temp)
    if not os.path.exists(CC_dir_temp):
      call(["mkdir", "-p", CC_dir_temp])

    badNodesPsisTaufile_pd = '{}badNodesPsisTauFile_PD_{}'.format(CC_dir_temp,currPrD)


    #badNodesPsisTau = dataR['badNodesPsisTau']
    #NodesPsisTauIQR = dataR['NodesPsisTauIQR']
    #NodesPsisTauVals = dataR['NodesPsisTauVals']
    #print ('read badNodesPsisTau', badNodesPsisTau,len(badPsis))
    #print ('read NodesPsisTauIQR',NodesPsisTauIQR[0:10,:])
    #if len(badPsis)>0:
    badNodesPsisTau = np.copy(badPsis)
    #print ('tauPsisIQR',tauPsisIQR,np.shape(tauPsisIQR))
    NodesPsisTauIQR = tauPsisIQR
    NodesPsisTauOcc = tauPsisOcc
    NodesPsisTauVals = tauPrDPsis


    time.sleep(2)
    myio.fout1(badNodesPsisTaufile_pd,['badNodesPsisTau','NodesPsisTauIQR' ,'NodesPsisTauOcc','NodesPsisTauVals'], [badNodesPsisTau,NodesPsisTauIQR,NodesPsisTauOcc, NodesPsisTauVals])
    time.sleep(2)
    #except:
    #    print('badNodes File: ',badNodesPsisTaufile,', does not exist.')

    computeOF = 1
    if computeOF:
        #calculate OF for each psi-movie
        #loop over for psi selections for current prD
        for psinum_currPrD in psiSelcurrPrD:
            IMGcurrPrD = moviePrDPsi[psinum_currPrD]

            #print('Current-PrD:{}, Current-PrD-Psi:{}'.format(currPrD, psinum_currPrD))
            prds_psinums = [currPrD, psinum_currPrD]
            FlowVecPrDPsi = ComputePsiMovieOpticalFlow(IMGcurrPrD,p.opt_movie,prds_psinums)
            FlowVecPrD[psinum_currPrD] =  FlowVecPrDPsi

        #print('Writing OpticalFlow-Node {} data to file\n\n'.format(currPrD))
        CC_OF_file = '{}'.format(CC_OF_file)
        myio.fout1(CC_OF_file,['FlowVecPrD'],[FlowVecPrD])
        #return FlowVecPrD

# If computing for a specified set of nodes, then call the function with nodeRange
def op(nodeEdgeNumRange, *argv):
    time.sleep(5)
    set_params.op(1)
    '''
    #set_params.op(-1)
    if not os.path.exists(p.CC_OF_dir):
        from subprocess import call
        call(["mkdir", "-p", p.CC_OF_dir])
    '''

    multiprocessing.set_start_method('fork', force=True)

    if argv:
        progress5 = argv[0]

    nodeRange = nodeEdgeNumRange[0]
    edgeNumRange = nodeEdgeNumRange[1]
    numberofJobs = len(nodeRange) + len(edgeNumRange)

    p.findBadPsiTau = 1 # This needs to be interfaced with the GUI , to find bad psi movies
    if p.findBadPsiTau:
        #initialize and write to file badpsis array
        #print('\nInitialize and write a file to record badPsis')
        offset_OF_files = len(nodeRange) - count1(nodeRange)
        #print('offset_OF_files',offset_OF_files)
        if offset_OF_files == 0: # offset_OF_files=0 when no OF files were generated
            badNodesPsisTaufile = '{}badNodesPsisTauFile'.format(p.CC_dir)
            if os.path.exists(badNodesPsisTaufile):
                os.remove(badNodesPsisTaufile)

        CC_graph_file_pruned = '{}_pruned'.format(p.CC_graph_file)
        if os.path.exists(CC_graph_file_pruned):
            dataG = myio.fin1(CC_graph_file_pruned)
        else:
            dataG = myio.fin1(p.CC_graph_file)

        G = dataG['G']
        badNodesPsisTau = np.zeros((G['nNodes'],p.num_psis)).astype(int)
        NodesPsisTauIQR = np.zeros((G['nNodes'],p.num_psis))+5. # any positive real number > 1.0 outside tau range
        # tau range is [0,1.0], since a zero or small tau value by default means it will be automatically assigned
        # as a bad tau depending on the cut-off
        NodesPsisTauOcc = np.zeros((G['nNodes'],p.num_psis))
        NodesPsisTauVals = [[None]]*G['nNodes']

        # the above variables are initialized at the start and also at resume of CC step
        # and used later for combining the individual bad tau PD files

        # but make sure the intialized variables are written out to the file only at the start
        # and not during resume of CC step
        if offset_OF_files==0:
            #print('badNodesPsisTaufile variables initialized...')
            myio.fout1(badNodesPsisTaufile, ['badNodesPsisTau','NodesPsisTauIQR','NodesPsisTauOcc' ,'NodesPsisTauVals'], [badNodesPsisTau,NodesPsisTauIQR,NodesPsisTauOcc, NodesPsisTauVals])
            #print('badNodesPsisTaufile initialized...')
            time.sleep(5)

    if p.machinefile:
        print('using MPI with {} processes'.format(p.ncpu))
        Popen(["mpirun", "-n", str(p.ncpu), "-machinefile", str(p.machinefile),
              "python", "modules/CC/ComputeOpticalFlowPrDAll_mpi.py"],close_fds=True)
        if argv:
            progress5 = argv[0]
            offset = 0
            while offset < len(nodeRange):
                offset = len(nodeRange) - count1(nodeRange)
                progress5.emit(int((offset / float(numberofJobs)) * 100))
                time.sleep(15)
    else:

        input_data = divide1(nodeRange) # changed Nov 30, 2018, S.M.
        if argv:
            offset = len(nodeRange) - len(input_data)
            #print('optical offset',offset)
            progress5.emit(int((offset / float(numberofJobs)) * 100))

        if p.ncpu == 1:  # avoids the multiprocessing package
            for i in range(len(input_data)):
                ComputeOptFlowPrDPsiAll1(input_data[i])
                if argv:
                    offset += 1
                    progress5.emit(int((offset / float(numberofJobs)) * 100))
        else:
            with poolcontext(processes=p.ncpu,maxtasksperchild=1) as pool:
                for i, _ in enumerate(pool.imap_unordered(partial(ComputeOptFlowPrDPsiAll1), input_data),1):
                    if argv:
                        offset += 1
                        progress5.emit(int((offset / float(numberofJobs)) * 100))
                    time.sleep(0.05)
                pool.close()
                pool.join()


    # multiprocessing is having difficulty in writing to the same file,
   # for now individual files were written and are being combined here
    if p.findBadPsiTau:
        CC_dir_temp = '{}temp/'.format(p.CC_dir)

        # if CC_dir_temp exists and is non-empty  combine the individual files again
        if os.path.exists(CC_dir_temp) and len(os.listdir(CC_dir_temp))>0:
            for currPrD in nodeRange:
                badNodesPsisTaufile_pd = '{}badNodesPsisTauFile_PD_{}'.format(CC_dir_temp,currPrD)
                dataR = myio.fin1(badNodesPsisTaufile_pd)
                time.sleep(1)
                #print( currPrD , dataR)
                badPsis = dataR['badNodesPsisTau'] # based on a specific tau-iqr cutoff in LoadPrDPsiMoviesMasked
                # but we actually use the raw iqr values to get a histogram of all iqr across all PDs to get the better cutoff later.
                tauPsisIQR = dataR['NodesPsisTauIQR']
                tauPsisOcc = dataR['NodesPsisTauOcc']
                tauPrDPsis = dataR['NodesPsisTauVals']
                #print ('read badNodesPsisTau', badNodesPsisTau,len(badPsis))
                #print ('read NodesPsisTauIQR',NodesPsisTauIQR[0:10,:])
                if len(badPsis)>0:
                    badNodesPsisTau[currPrD,np.array(badPsis)] = -100
                #print('tauPsisIQR',np.shape(tauPsisIQR))
                NodesPsisTauIQR[currPrD,:] = tauPsisIQR
                NodesPsisTauOcc[currPrD,:]= tauPsisOcc
                NodesPsisTauVals[currPrD] = tauPrDPsis

            badNodesPsisTaufile = '{}badNodesPsisTauFile'.format(p.CC_dir)
            myio.fout1(badNodesPsisTaufile, ['badNodesPsisTau','NodesPsisTauIQR','NodesPsisTauOcc' ,'NodesPsisTauVals'], [badNodesPsisTau,NodesPsisTauIQR,NodesPsisTauOcc,NodesPsisTauVals])

            rem_temp_dir=0
            if rem_temp_dir:
                # remove the temp directory if rem_temp_dir=1, or manually delete later
                print('Removing temp directory',CC_dir_temp)
                import shutil
                if os.path.exists(CC_dir_temp):
                        shutil.rmtree(CC_dir_temp)
