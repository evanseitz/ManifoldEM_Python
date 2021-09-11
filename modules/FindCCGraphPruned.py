import numpy as np
import logging, sys
#from subprocess import call
sys.path.append('./CC/')
sys.path.append('../')
import myio
#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

import p

from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse import tril,triu
from scipy.io import loadmat
from scipy.sparse.csgraph import connected_components

'''	
Copyright (c) Columbia University Suvrajit Maji 2019    	
'''

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

'''
function
G = CreateGraphStruct(Xp, AdjMat, nStates, pwDist, numNbr, epsilon)
% Create graph structure
% Input:
% Xp: 3xn starting points(coordinates) on S2, n is the number of nodes
% nStates: number of states each node can have
% Note:: we assume that all nodes have same number of states here.
% numNbr: number of nearest neighbor we would like to have for each
% neighbor
% Output:
% G: graph structure
%
%
% Suvrajit Maji, sm4073 @ cumc.columbia.edu
% Columbia University
% Created: Feb 02, 2018. Modified:Jan 25, 2019
Python version Hstau Liao copyright 2018
'''

def CreateGraphStruct(nStates, pwDist, epsilon,*argv):

    '''try:
        assert (len(argv) > 0)
    except AssertionError:
        _logger.error('wrong nmber of arguments')
        _logger.exception('wrong nmber of arguments')
        raise
        sys.exit(1)
    '''
    if type(pwDist) is list:
        pwDist=np.array(pwDist)

    if argv:
        #print '\nAdjMat is available to create the graph.'
        AdjMat = argv[0]
    else:
        #print '\nAdjMat is not available.'
        AdjMat = np.empty(0)

    if pwDist.shape[0] > 0:
        nNodes = pwDist.shape[0]

    elif AdjMat.shape[0]> 0:
        nNodes = AdjMat.shape[0]
    else:
        #print '\npwDist and AdjMat both cannot be empty.'
        return -1


    #print pwDist[0:10][:,0:10]

    Nodes = range(nNodes)
    G = dict(nNodes=nNodes,Nodes=Nodes)
    #print 'Number of Graph Nodes:', nNodes
    # create state for each node
    if np.isscalar(nStates):
        # isscalar(nStates) # if it is a scalar then it is equal to maxState
        nStates = nStates * np.ones(nNodes,dtype='int')
        G.update(eqnStates=1)
    else:
        G.update(eqnStates=0)

    G.update(nStates = nStates)
    maxState = np.max(nStates)
    G.update(maxState = maxState)

    nnMat = np.empty((G['nNodes'],),dtype=object)
    #nnMat =[] # default if not doing the nn search
    if not argv: # adj matrix absent
        print('Using pwDist to create the graph (AdjMat).')
        # create the connections from neighbor search
        Adj = (pwDist <= epsilon) * (pwDist!=0)
        Adj = csr_matrix(Adj)
        # form the graph model
        for n in range(nNodes):
            #print 'nnMat',type(nnMat)
            nnMat[n] = np.nonzero(Adj[n,:])[1]

        # if it is not symmetric
        Adj = Adj + Adj.T
        AdjMat = (Adj > 0)

    else:
        for n in range(nNodes):
            #print 'nnMat',type(nnMat)
            nnMat[n] = np.nonzero(AdjMat[n,:])[1]

    AdjMat = csr_matrix(AdjMat)

    # 1.
    # create edge indices
    ni, nj = np.nonzero(AdjMat)

    Edges = np.vstack((nj,ni)).T
    #I = np.argsort(Edges[:,[0,1]])
    #Edges=Edges[I,:]

    I = np.lexsort((Edges[:,1],Edges[:,0]))
    Edges=Edges[I,:]
    #print "Edges.shape", Edges.shape

    #print 'Remove the reverse links'
    Edges = Edges[np.nonzero(Edges[:, 0] < Edges[:, 1])]
    #print 'Number of Graph Edges:', Edges.shape


    # 2.
    ni, nj = np.nonzero(tril(AdjMat))

    ### to make the output same as matlab implementation
    nij = np.c_[ni,nj]
    nids = np.lexsort((nij[:,0],nij[:,1]))
    nij_s = nij[nids,:]
    ni = nij_s[:,0]
    nj = nij_s[:,1]

    nEdges = len(ni)
    val_e = np.arange(nEdges)+1 # for now, to compare with matlab we need EdgeIdx to contain 1 to nEdges even for python

    Ni = np.hstack((ni,nj)).T
    Nj = np.hstack((nj, ni)).T
    Val = np.hstack((val_e,val_e+nEdges)).T
    EdgeIdx = csr_matrix((Val, (Ni, Nj)), shape=AdjMat.shape)

    G.update(Nodes=Nodes, nNodes=nNodes,epsilon=epsilon,
             nnMat=nnMat, AdjMat=AdjMat, Edges=Edges,
             nEdges=nEdges, EdgeIdx=EdgeIdx, nStates=nStates)
    return G


'''
function [Gsub,G] = getSubGraph(G,nodes)
% Obtain subgraph of a graph G given the subset of nodes to be included
% Input:
%   G: Input graph / adjacency
%   nodes: subset of nodes to be included into the subgraph
%
% Output:
%   Gsub: subgraph, returns cell array of graph structure(s) for multiple
%   subgraphs
%
%
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Mar 12,2018. Modified: Jan 25,2019
%
%
'''
def getSubGraph(G,*nodes):
    print("\nPerforming connected component analysis.")
    A = G['AdjMat']
    S,C = connected_components(A)
    G.update(NodesConnComp=[],AdjConnComp=[])
    for i in np.arange(S):
        idxc = np.nonzero(C==i)[0]
        G['NodesConnComp'].append(idxc)
        G['AdjConnComp'].append(A[idxc][:, idxc])


    numConnComp = len(G['NodesConnComp'])
    print("Number of connected components (with isolated node(s)):",numConnComp)

    Gsub = []
    if not nodes:
        # get all subgraphs
        for i in range(numConnComp):
            #print '\nConnected Component',i,':',G['NodesConnComp'][i], ', size:',len(G['NodesConnComp'][i])
            if len(G['NodesConnComp'][i]) == 1:
                #print 'Singlet Node in connected component',i
                sn=1 # do nothing
            nodes = G['NodesConnComp'][i]
            if hasattr(G,'AdjConn'):
                Asub = G['AdjConnComp'][i]
            else:
                Asub = G['AdjMat'][nodes][:,nodes]

            Gsub.append(CreateGraphStruct(G['maxState'],[],G['epsilon'], Asub))

            # nnMat with the nodes in the subgraph only
            Gsub[i]['nnMat'] = np.array(G['nnMat'])
            Gsub[i]['originalNodes'] = nodes
            einds = np.in1d(G['Edges'][:,0],nodes) | np.in1d(G['Edges'][:,1],nodes)
            Gsub[i]['originalEdgeList'] = np.nonzero(einds)
            Gsub[i]['originalEdges'] = G['Edges'][einds,:]

    else:
        # get the subgraph with the specified nodes only
        Asub = G['AdjMat'][nodes][:,nodes]
        Gsub = CreateGraphStruct(G['MaxState'],[],[],Asub)
        Gsub['originalNodes'] = nodes
        einds = np.in1d(nodes, G['Edges'][:,0]) or np.in1d(nodes, G['Edges'][:,1])
        Gsub['originalEdgeList'] = np.nonzero(einds)
        Gsub['originalEdges'] = G['Edges'][einds,:]


    return (Gsub, G)



def CalcPairwiseDistS2(X,*argv):
    '''
    [pwDotProd, pwDist] = CalcPairwiseDistS2(X, prD1, prD2)
    pairwise projection angular - distance and Euclidean distance calculations
    Input:
        X: 3xN coordinate matrix
        prD1: point 1 out of 1...N
        prD2: point 2 out of 1...N
    Output:
        pwDotProd: dot product(angular distance) between the points
        pwDist: Euclidean distance between the points

    Author Suvrajit Maji, sm4073@cumc.columbia.edu
    Columbia University
    Created: Dec 2017.
    Modified: Feb02, 2018

    Python version Hstau Liao copyright 2018
    '''
    try:
        assert (len(argv) == 0 or len(argv) == 2)
    except AssertionError:
        _logger.error('wrong nmber of arguments')
        _logger.exception('wrong nmber of arguments')
        raise
        sys.exit(1)

    if not argv:
        UIdxs = np.arange(X.shape[1])
        VIdxs = np.arange(X.shape[1])
    else:
        UIdxs = argv[0]
        VIdxs = argv[1]
    #For two sets of matrices U & V
    U = X[:, UIdxs]
    V = X[:, VIdxs]

    # pairwise dot product
    pwDotProd = np.dot(U.T,V)

    # pairwise Euclidean distance
    Dsq = np.sum(U*U,axis=0).T + np.sum(V*V,axis=0) - 2*np.dot(U.T , V)
    #Dsq[Dsq < 0] = 0
    Dsq[Dsq < 1e-6] = 0
    pwDist = np.sqrt(Dsq)
    return (pwDotProd, pwDist)


def op(CC_graph_file_pruned):
    import p

    trash_list=np.array(p.trash_list)
    good_nodes = np.nonzero(trash_list==0)[0]
    bad_nodes = np.nonzero(trash_list==1)[0]
    numNodes = len(good_nodes)
    num_pruned_nodes = len(bad_nodes)

    print("Number of isolated nodes in the graph after pruning:",num_pruned_nodes)

    maxState = 2 * p.num_psis # Up and Down

    if numNodes> 1:

        # Modifying the graph structure
        data = myio.fin1(p.CC_graph_file)
        G=data['G']
        epsilon = G['epsilon'] # save it later after update
        G.update(nPsiModes=p.num_psis)

        print('Number of Graph Edges before prunning:', G['nEdges'])
        # prune edges corresponding to the bad nodes with actually removing those bad nodes by disconnecting the edges
        # in and out of those specified nodes
        newAdjMat=lil_matrix(G['AdjMat'])
        newAdjMat[bad_nodes,:]=0
        newAdjMat[:,bad_nodes]=0


        newAdjMat = csr_matrix(newAdjMat)
        G.update(AdjMat=newAdjMat)
         # Updated graph info
        #G = CreateGraphStruct(G['maxState'],[],[], newAdjMat)
        G = CreateGraphStruct(G['maxState'],[],[], newAdjMat) # june 2020


        # re-insert the epsilon
        G['epsilon'] = epsilon

        print('Number of Graph Edges after pruning:', G['nEdges'])
        # If the min distance /epsilon values are different during initial and pruned graph creation
        # the actual number of edges may or may not decrease after pruning, the nodes that were pruned will become
        # isolated

        #print 'bad_nodes',bad_nodes
        #print 'Number of Nodes in the pruned graph',np.unique(G['Edges'])


    else:
        print('Single PrD. Empty graph structure created with one node.')
        G = CreateGraphStruct(maxState, [0], 0)

    # Determine if there are multiple connected components / subgraph
    # proceed to the pairwise measurements only after we are fine with the connected components
    Gsub, G = getSubGraph(G)

    # if pruned
    # replace
    #CC_graph_file_pruned = '{}_pruned'. format(p.CC_graph_file)
    myio.fout1(CC_graph_file_pruned,['G', 'Gsub'],[G, Gsub])

    return G,Gsub


if __name__ == '__main__':
    import p, os
    p.init()
    p.user_dir = '../'
    p.out_dir = os.path.join(p.user_dir, 'data_output/')
    p.tess_file = '{}/selecGCs'.format(p.out_dir)
    p.nowTime_file = os.path.join(p.user_dir, 'data_output/nowTime')
    p.create_dir()
    op()
