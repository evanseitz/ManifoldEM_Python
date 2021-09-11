import numpy as np
import logging, sys

sys.path.append('../')

import myio
import p
import datetime
import ComputeOpticalFlowPrDAll
import ComputeMeasureEdgeAll
import FindCCGraphPruned
from subprocess import call
import os
import time

#from pyface.qt import QtGui, QtCore
#os.environ['ETS_TOOLKIT'] = 'qt4'

'''	
Copyright (c) Columbia University Suvrajit Maji 2019	
'''

# this rescaling function should ensure to keep the exp(-M) values within a certain range as to prevent
# numerical overflow/underflow
def reScaleLinear(M,edgeNumRange,mvalrange):

    numE = np.max(edgeNumRange)#M.shape[0]
    nm = np.zeros(numE+1).astype(int)
    all_m=[]
    print('edgeNumRange',np.shape(edgeNumRange))
    for e in edgeNumRange:
        #print 'scale e',e
        meas = M[e].flatten()
        #print meas.shape
        nm[e] = meas.shape[0]
        all_m.append(meas)

    all_m = np.squeeze(all_m).flatten()

    #print 'min',np.min(all_m),'max',np.max(all_m)

    # determine if there are outliers in the all_m array
    #mean_val = np.mean(all_m)
    #sd_val = np.std(all_m)
    #upper_thresh = mean_val+3*sd_val

    q1, q3 = np.percentile(all_m,[25,75])
    iqr = q3 - q1
    upper_thresh =  q3 + (1.5 * iqr)

    #print 'upper_thresh',upper_thresh
    #outlier_ids = all_m>upper_thresh
    #all_m[outlier_ids]= upper_thresh
    #print 'max-without outliers',np.max(all_m[all_m<=upper_thresh])
    all_m[all_m>upper_thresh]= upper_thresh
    #print all_m
    ## linear scaling of values within the range 'mvalr', min and max to mapped to min(mvalr) and max(mvalr)
    scaled_all_m = np.interp(all_m,(np.min(all_m), np.max(all_m)),mvalrange)

    M_scaled = np.empty(M.shape,dtype=object)
    #print  M_scaled.shape
    for e in edgeNumRange:
        if e==0:
            nm_ind = range(0,nm[e])
        else:
            nm_ind = range(e*nm[e-1],e*nm[e-1]+nm[e])

        M_scaled[e]= np.reshape(scaled_all_m[nm_ind],np.shape(M[0]))

        #print 'e',e
        #print 'meas:',M_scaled[e]
    return M_scaled


def findThreshHist(X,nbins,method=1,vis=1):
    # this is still experimental
    # would work if the data values are sort of bi-modal distribution
    import matplotlib.pyplot as plt

    def findIntersectionOfFuncs(funce, funcg, popt, x0):
        from scipy.optimize import fsolve
        p1 = popt[0]
        p2 = popt[1]
        p3 = popt[2]
        p4 = popt[3]
        p5 = popt[4]
        sol_root = fsolve(lambda x: funce(x, p1, p2) - funcg(x, p3, p4, p5), x0)
        return sol_root

    def separateHist(X,labels,cluster_centers,bedges,tl,vis):
        id0 = np.argmin(cluster_centers)
        id1 = np.argmax(cluster_centers)
        Xid0=X[labels == id0]
        Xid1=X[labels == id1]
        if vis:
            plt.hist(Xid0,bedges,color='gold')
            plt.hist(Xid1,bedges,color='green')
            plt.title(tl,fontsize=20)
            plt.show()
        th = np.max(X[labels==id0])
        return th

    # histogram
    h, bedges = np.histogram(X, bins=nbins)
    bctrs = bedges[:-1] + np.diff(bedges) / 2.0


    if method==0 or method==1 or method==4 or method=='all':
        # 1. kmeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2).fit(X)
        if method==1:
            vis=0
        t_thresh_k = separateHist(X, kmeans.labels_, kmeans.cluster_centers_.T,bedges,'kmeans',vis)
        if method==0 or method=='all':
            print('0. Kmeans.threshold:',t_thresh_k,', centers:',kmeans.cluster_centers_.T)

        t = t_thresh_k.copy()

    if method==1 or method=='all':
        # 2. find_peaks -- valleys for 'inverted' data
        # somehow argrelextrema did not produce all correct extremas for different data
        # so sticking with find_peaks
        # TO DO. If there are multiple valleys found, then choose the one in between the two high-peaks found using
        # yfilt and not -yfilt
        # or other methods such as kmeans (method=1) ... just have to eliminate the valley (inverted peaks) indexes
        # outside the the two peak values...(assumption is there are only two high peaks...for multiple high peaks,
        # there will be different criteria)
        from scipy import signal
        from scipy.signal import find_peaks
        import scipy.ndimage.filters as ndifilt
        yfilt = ndifilt.uniform_filter1d(h, 4, mode='nearest')
        p, _= find_peaks(-yfilt, prominence=nbins//2)
        valleys = bctrs[p]
        indx=np.where(np.logical_and(valleys >=np.min(kmeans.cluster_centers_), valleys <= np.max(kmeans.cluster_centers_)))
        t_thresh_p = valleys[indx]

        print('1. find-peaks(inverted).threshold:',t_thresh_p,'peaks:',bctrs[p])
        #if there are multiple peaks/valleys, choose the one between the centers found by other reliable methods
        # such as kmeans below
        t = t_thresh_p.copy()

    if method==2 or method=='all':
        # 3. otsu
        from skimage import filters
        t_thresh_o = filters.threshold_otsu(X)
        print('2. Otsu.threshold:', t_thresh_o)
        t = t_thresh_o


    if method==3 or method=='all':
        # works best if the multimodal distributions are all mixture of gaussians
        #4. GMM
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=2,covariance_type = "full")
        gmm.fit(X)
        stdevs = np.sqrt(gmm.covariances_).T # if we have the cov matrix, then sdev=np.sqrt(np.diag(cov))
        #print(stdevs)
        # TODO: if mixture does not seem to contain both gaussians
        # so, fit separate mixture of two different distributions (e.g. exponential + gaussian)
        mthresh=0.2 # check this based on stdevs ??
        if np.max(np.sqrt(gmm.covariances_))< mthresh:
            print('... Gmm: individual covariance did not work well (mixture is possibly not all gaussians)... using ''tied'' covariance for gmm.')
            gmm = GaussianMixture(n_components=2,covariance_type = "tied")
            gmm.fit(X)

        glabels = gmm.predict(X)
        t_thresh_g = separateHist(X, glabels, gmm.means_.T,bedges,'gmm',vis)
        print('3. Gmm.threshold:',t_thresh_g,'centers:',gmm.means_.T)
        t= t_thresh_g.copy()

    if method==4 or method=='all':
        #5. This for only if we have exp+gauss (given the data) ...
        # a. If the data has only mixture of gaussians, then use gauss+gauss curve fit,
        # in fact gmm(method=4) would work fine in that case and curve fit would not be required.
        # b. Uses filtered h.
        # c. Also uses kmeans results (method=1) as intial guess for gauss center
        # Note: if for some reason , the curve-fitting fails, we use the default kmeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2).fit(X)

        def funce(x,ea,eb):
            return ea*np.exp(-eb * x)

        def funcg(x,ga,gb,gc):
            return ga*np.exp(-((x-gb)/gc)**2)

        def func(x,ea,eb,ga,gb,gc,f):
            return funce(x,ea,eb) + funcg(x,ga,gb,gc) + f

        p0 = [np.max(h), 1.0, np.max(h)/2, np.max(kmeans.cluster_centers_), 1.0,0.0]

        from scipy.signal import savgol_filter
        xdata = bctrs
        ydata = savgol_filter(h,7,2)

        try:
            # fitting may not always converge or work
            popt, pcov = curve_fit(func, xdata, ydata, p0)
            t_thresh_c = findIntersectionOfFuncs(funce, funcg, popt, 0.0)
            print('4. Curve_fit intersection.threshold:',t_thresh_c,', centers:',kmeans.cluster_centers_.T)
            t = t_thresh_c.copy()

            if vis:
                plt.plot(xdata, ydata, 'b-', label='data')
                plt.plot(xdata, func(xdata, *popt), '-', color='r', lw=2)
                plt.plot(xdata, funce(xdata, *popt[0:2]), '-', color='orange', lw=3)
                plt.plot(xdata, funcg(xdata, *popt[2:5]), '-', color='green', lw=3)
                plt.plot(t_thresh_c, funcg(t_thresh_c, *popt[2:5]), 'ro', markersize=8)
                plt.title('curve_fit intersection',fontsize=20)
                plt.show()

        except:
            print('4. curve-fitting failed..., using kmeans instead.')
            # in that case just use the kmeans values (method=1)?
            #t_thresh_c = separateHist(X, kmeans.labels_, kmeans.cluster_centers_.T,bedges,'kmeans',0)
            t_thresh_c = t_thresh_k.copy()
            print('Kmeans.threshold:',t_thresh_c,', centers:',kmeans.cluster_centers_.T)
            t = t_thresh_c.copy()


    if method=='all':
        t = np.append(t_thresh_k,np.append(t_thresh_p, [t_thresh_o , t_thresh_g]))
        t = np.append(t, t_thresh_c)

    print('\nDistribution cutoff(s):',t)
    return t, np.max(h)



def checkBadPsis(trash_list):
    ### Oct 2020, this is still experimental
    ### It would be good to interface this part also with the GUI , to visually check the cut-off selected
    ### for the bad tau iqr distribution ...
    ### Check if there are significant bad PDs(>10) after Optical Flow computations of the psi-movies,
    # if yes then prune those bad nodes. The nodes are not removed from the graph but the edges are modified
    # just update the graph G with new edge connections

    # the bad psis are also checked during BP for setting tiny node potentials for bad node states
    # it is there in case, the threshold needs to be modified to exclude or include more bad psi-movies
    #
    # the graph edges can be pruned here
    import matplotlib.pyplot as plt

    badNodesPsisTaufile = '{}badNodesPsisTauFile'.format(p.CC_dir)
    dataR = myio.fin1(badNodesPsisTaufile)

    badNodesPsisTau = dataR['badNodesPsisTau'] # this was pre-calculated using some tau-cutoff, here we are update it
    TausCell=dataR['NodesPsisTauVals']
    TausMat=dataR['NodesPsisTauIQR']

    #print 'TausMat', TausMat[0:10,:]
    print('Using all psi-tau values across all nodes to find the tau(iqr) distribution-cutoff')
    TausAll = TausMat.flatten()
    X = TausAll.reshape(-1,1)

    #tau_thresh = findThreshHist(X)

    nbins=60 # we could use optimal bin finding methods such as 'fd','scott' etc,
    #plt.savefig('tau_iqrhist_cutoff.png')
    visual=0

    Allmethods = ['K-means','find_peaks','Otsu','GMM','Curve-fit Intersection']
    # choose cutoff method type
    # 0. Kmeans
    # 1: find_peaks
    # 2: Otsu
    # 3. GMM (this works really well when the individual distributions are all gaussians,
    #        otherwise GMM will have issues)
    # 4. Curve fitting with Kmeans

    method=1#'all' #integer between 0 to 4 or 'all'
    numAllMethods = len(Allmethods)

    if method == 'all':
        methods = range(numAllMethods)
        print('Method:', Allmethods)
    else:
        methods=[]
        methods.append(method)
        print('Method:', Allmethods[method])

    cutoff,hmax = findThreshHist(X,nbins,method=method,vis=visual)

    if visual:
        colors=['r','orange','b','brown', 'olive']
        plt.hist(X,nbins)
        ymin=0;ymax=hmax
        legends=[]
        for i in methods:
            #print('i',i)
            color=colors[i]
            if method == 'all':
                cutf=cutoff[i]
            else:
                cutf=cutoff
            #print(color)
            plt.vlines(cutf,ymin,ymax,color=color,linestyle='dashed',lw=3)
            legends.append(Allmethods[i])
        ### use this to visually check the additional cut-off using mean, median etc. of all cutoffs.
        if method == 'all': # what to do when using multiple methods , if we want a single cut-off ?
            plt.vlines(np.median(cutoff),ymin,ymax,color='green',linestyle='dashed',lw=3)

        plt.legend(legends,loc='upper right',fontsize=20)
        plt.title('Tau distribution cutoff(s)',fontsize=20)
        plt.show()

    if len(cutoff) > 1: # how do we compare and choose the cut-off when using multiple methods?
                        # just choose the min, max, mean , median , etc. oro compare
                        # the cut-off lines visually and just choose the best one ?
        #tau_thresh = np.median(cutoff)
        best_id = 1 #? visually compare [0...4]
        tau_thresh = cutoff[best_id]
    else:
        tau_thresh = cutoff.copy()

    print('Tau(iqr) distribution cutoff selected:',tau_thresh)
    if len(tau_thresh)>0:
        bad_idx = (TausMat<tau_thresh)
    else:
        bad_idx=[]
    badNodesPsisTau = np.zeros(np.shape(TausMat))
    badNodesPsisTau[bad_idx]=-100

    extra = dict(badNodesPsisTau_of=badNodesPsisTau)
    dataR.update(extra)
    #print 'dataR',dataR['badNodesPsisTau_of']

    #previously was generating a separate file *_of but now just adding an extra variable to the same file
    #badNodesPsisTau_of which should used at BP step
    badNodesPsisTaufile_of = '{}'.format(badNodesPsisTaufile)
    myio.fout2(badNodesPsisTaufile_of,dataR)

    # sort of redundant to set -100 and not bad_idx directly but for consistency, we leave it like this for now
    NumbadPsis = np.sum(badNodesPsisTau==-100,axis=1)
    NumPsis = badNodesPsisTau.shape[1]
    nodesAllBadPsis = np.nonzero(NumbadPsis==NumPsis)[0]
    num_nodesAllBadPsis = len(nodesAllBadPsis)
    print('Number of trash PDs detected using auto tau-cutoff:',num_nodesAllBadPsis)

    np.savetxt('{}NodeTauPsis_of.txt'.format(p.CC_dir), TausMat, fmt="%f", newline="\n")
    np.savetxt('{}badNodePsis_of.txt'.format(p.CC_dir), badNodesPsisTau, fmt="%d", newline="\n")
    np.savetxt('{}nodesAllBadPsis_of.txt'.format(p.CC_dir), nodesAllBadPsis+1, fmt="%d", newline="\n")

    if nodesAllBadPsis.shape[0]>0:
        trash_list[nodesAllBadPsis]=int(1)

    return trash_list,num_nodesAllBadPsis


def op(G, nodeRange, edgeNumRange, *argv):

    nodeEdgeNumRange = [nodeRange,edgeNumRange]

    # Step 1. Compute Optical Flow Vectors
    # Save the optical flow vectors for each psi-movie of individual projection direction
    if p.getOpticalFlow:
        print('\n1.Now computing optical flow vectors for all (selected) PrDs...\n')
        #Optical flow vectors for each psi-movies of each node are saved to disk
        if argv:

            ComputeOpticalFlowPrDAll.op(nodeEdgeNumRange,argv[0])
        else:
            ComputeOpticalFlowPrDAll.op(nodeEdgeNumRange)



    ## check for bad PDs found based on bad tau values
    trash_list = p.trash_list
    # take the already existing trash_list and update it
    trash_list_chk, num_nodesAllBadPsis = checkBadPsis(trash_list)

    # trash_list_chk will be used inside the following pruned graph creation if p.use_pruned_graph =1
    # p.trash_list = trash_list_chk
    # FindCCGraphPruned uses p.trash_list to create the pruned graph

    CC_graph_file_pruned = '{}_pruned'. format(p.CC_graph_file)

    p.use_pruned_graph = 0 # interface with gui

    if p.use_pruned_graph:
        #Step 2a. June 2020
        ### Check if there are significant bad PDs(>10) after Optical Flow computations of the psi-movies,
        # if yes then prune those bad nodes. The nodes are not removed from the graph but the edges are modified
        # just update the graph G with new edge connections
        #the bad psis are also checked during BP for setting tiny node potentials for bad node states
        # the graph edges can be pruned here

        num_bad_nodes_prune_cutoff=5
        print('Pruning the graph G if there are more than {} bad nodes'.format(num_bad_nodes_prune_cutoff))

        # update the p.trash_list
        p.trash_list = trash_list_chk

        if num_nodesAllBadPsis>num_bad_nodes_prune_cutoff:

            if not os.path.exists(CC_graph_file_pruned):
                 G,Gsub = FindCCGraphPruned.op(CC_graph_file_pruned)
            else:
                print('Using a previously pruned graph.')
                data = myio.fin1(CC_graph_file_pruned)
                G=data['G']
                Gsub = data['Gsub']
            numConnComp = len(G['NodesConnComp'])
            #print "Number of connected component:",numConnComp

            anchorlist = [a[0] for a in p.anch_list]
            anchorlist = [a - 1 for a in anchorlist] # we need labels with 0 index to compare with the node labels in G, Gsub


            nodelCsel = []
            edgelCsel = []
            # this list keeps track of the connected component (single nodes included) for which no anchor was provided
            connCompNoAnchor = []
            for i in range(numConnComp):
                nodesGsubi = Gsub[i]['originalNodes']
                edgelistGsubi = Gsub[i]['originalEdgeList']
                edgesGsubi = Gsub[i]['originalEdges']
                #print 'Checking connected component ','i=',i,', Gsub[i]',', Original node list:',nodesGsubi,\
                #  'Original edge list:',edgelistGsubi[0],'Original edges:', edgesGsubi, 'No. edges:',len(edgesGsubi)

                #print 'Checking connected component ','i=',i,', Gsub[i]',', Original node list:',nodesGsubi,\
                #    'Original edge list:',edgelistGsubi[0],'Size Edges:',len(edgesGsubi)

                if any(x in anchorlist for x in nodesGsubi) or len(nodesGsubi)>1:
                    #print 'Atleast one anchor node in connected component',i,'is selected.\n'
                    nodelCsel.append(nodesGsubi.tolist())
                    edgelCsel.append(edgelistGsubi[0])
                else:
                    connCompNoAnchor.append(i)
                    #print 'Anchor node(s) in connected component',i,' NOT selected.'
                    #print '\nIf you proceed without atleast one anchor node for the connected component',i,\
                    #    ', all the corresponding nodes will not be assigned with conformational coordinate labels.' \
                    #    'Cancel this program now or it will continue without the required anchors.\n'
                    #time.sleep(20)

            if len(connCompNoAnchor)>0:
                print('There are {} connected components with no anchors assigned. You can choose anchors for them ' \
                      'after the edge measurements are done, and re-run only the BP'.format(len(connCompNoAnchor)))

            #print connCompNoAnchor,G['ConnCompNoAnchor']


            nodeRange = [y for x in nodelCsel for y in x] #flatten list another way?
            edgeNumRange = [y for x in edgelCsel for y in x] #flatten list another way?

            nodeEdgeNumRange = [nodeRange,edgeNumRange]


            data = myio.fin1(CC_graph_file_pruned)
            extra = dict(nodeRange=nodeRange,edgeNumRange=edgeNumRange,ConnCompNoAnchor=connCompNoAnchor)
            data.update(extra)
            myio.fout2(CC_graph_file_pruned,data)



    # Step 2. Compute the pairwise edge measurements
    # Save individual edge measurements
    if p.getAllEdgeMeasures:
        print('\n2.Now computing pairwise edge-measurements...\n')
        # measures for creating potentials later on
        # edgeMeasures files for each edge (pair of nodes) are saved to disk
        if argv:
            ComputeMeasureEdgeAll.op(G,nodeEdgeNumRange,argv[0])
        else:
            ComputeMeasureEdgeAll.op(G,nodeEdgeNumRange)

    # Step 3. Extract the pairwise edge measurements
    # to be used for node-potential and edge-potential calculations
    print('\n3.Reading all the edge measurements from disk...')
    # load the measurements file for each edge separately
    #edgeMeasures = np.empty((len(edgeNumRange)),dtype=object)

    #in case there are some nodes/edges for which we do not want to calculate the measures,the number of edges and
    # max edge indices may not match, so use the full G.nEdges as the size of the edgeMeasures. The edges which are not
    # calculated will remain as empty

    print('Edges',G['nEdges'])
    edgeMeasures = np.empty((G['nEdges']),dtype=object)
    edgeMeasures_tblock = np.empty((G['nEdges']),dtype=object)
    badNodesPsisBlock = np.zeros((G['nNodes'],p.num_psis))
    for e in edgeNumRange:
        currPrD = G['Edges'][e,0]
        nbrPrD =  G['Edges'][e,1]
        #print 'Reading Edge:',e,'currPrD:',currPrD,'nbrPrD:',nbrPrD
        CC_meas_file = '{}{}_{}_{}'.format(p.CC_meas_file,e,currPrD,nbrPrD)
        data = myio.fin1(CC_meas_file)
        measureOFCurrNbrEdge = data['measureOFCurrNbrEdge']
        measureOFCurrNbrEdge_tblock = data['measureOFCurrNbrEdge_tblock']
        bpsi = data['badNodesPsisBlock']
        badNodesPsisBlock = badNodesPsisBlock + bpsi
        edgeMeasures[e] = measureOFCurrNbrEdge
        edgeMeasures_tblock[e] = measureOFCurrNbrEdge_tblock


    ###Test 30aug2019
    #print 'before:',edgeMeasures[0][:,:]
    #print 'before:',edgeMeasures[2][:,:]
    scaleRange = [5.0,50.0]
    edgeMeasures = reScaleLinear(edgeMeasures,edgeNumRange,scaleRange)
    #print 'badNodesPsisBlock',badNodesPsisBlock[0:10,:]
    #print 'after rescale:',edgeMeasures[0][:,:]
    #print 'after rescale:',edgeMeasures[2][:,:]
    #print 'badNodesPsis',badNodesPsis
    #print 'edgeMeasures_tblock',type(edgeMeasures_tblock)
    return edgeMeasures,edgeMeasures_tblock,badNodesPsisBlock
