import numpy as np
import copy
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

'''
function BPalg = createBPalg(G,options)
% create BPalg struct
% Input:
%   G: Input graph structure
%   options: options for belief propagation iterations
% Output:
%   BPalg: create BPalg structure
%
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Feb 10,2018. Modified:Feb 10,2018
Copyright (c) Columbia University Suvrajit Maji 2018 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
'''

def createBPalg(G,options):

    BPalg = dict(options=options,
                 G=G,
                 init_message = [], #% once set this is not modified for record
                 old_message = [],  #% this is updated with each iteration and is the last message before final iteration or convergence
                 new_message = [],  #% this is updated with each iteration and is the final message after the final iteration or convergence
                 nodeBel =[],
                 edgeBel =[],
                 alphaDamp = options['alphaDamp'],
                 eqnStates = options['eqnStates'],
                 bpError = [], #%sum(abs(BPalg.new_message - BPalg.old_message));
                 bpIter = 1,
                 convergence = 0, # % is set to 1 if converged, otherwise 0
                 convergenceIter = np.Inf)
    return BPalg


'''
function [N, z] = Normalize(M,dim)
% Normalize: Normalize a matrix M to N with the normalizing factor z
% Matrix entries sum to 1 along the specified dimension dim
% If no dimension is provided, use dim =1;
% if dim = 0, normalize the entire matrix, i.e. sum of all
% elements of the matrix is 1
% Usage:
% [N, z] = Normalize(M)
% [N, z] = Normalize(M,dim)
%
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Feb 06,2018. Modified:Feb 07,2018
Copyright (c) Columbia University Suvrajit Maji 2018 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
'''
def Normalize(M,*argv):
    if not argv:
        dim = 1
    else:
        dim=argv[0]

    if dim==0:
        z = sum(M.flatten())
    else:
        z = np.sum(M,axis=dim-1)

    #N = bsxfun(@rdivide,M,z)
    N = np.divide(M,z)

    return N



# max product
def max_product(A,x):
    #function y = max_product(A,x)
    #% max_product is like matrix multiplication, but sum gets replaced by max
    #% function y=max_mult(A,x) y(i) = max_j A(i,j) x(j)

    #% This is faster
    #print 'len(x.shape)',len(x.shape)
    if len(x.shape)==1:
        x = np.reshape(x,(len(x),1))   # convert a(r,) to a(r,1)
        # #print 'A',A.shape,'x.shape:',x.shape

    if x.shape[1]==1:
        X= np.matmul(x,np.ones((1,A.shape[0]))) #% X(i,j) = x(i)
        # #print 'X',X.shape
        y = np.max(A.T*X,axis=0).T #element wise multiplication At*X

    else: # this should be checked
        #%this works for arbitrarily sized A and x (but is ugly, and slower than above)
        X=np.matlib.repmat(x, (1, 1, A.shape[0]))
        B=np.matlib.repmat(A, (1, 1, x.shape[1]))
        C=np.transpose(B,[1,2,0])
        y=np.transpose(np.max(C*X,axis=0),[2,1,0])

    return y



'''
function [edgeIdxsUnDr,edgeIdxsDr,edgeIdxsRev] = getEdgeIdxsGivenNode(G,n)
%   Given a node, find all the edge(s) indices involving that node
%   (edgeIdxs lies between 1:nEdges for undirected and
%    between 1:2*nEdges for directed graph)
% Input:
%   G: graph structure
%   node: node involved in all edges to be found
%   edgeType:'undirected' or 'directed'
%   if edgeType is 'undirected', edgeIdx for n1--n2 and edgeIdx for n2--n1 will
%   be same although those edges may or may not be stored as two separate indices in G.EdgeIdx
%   if edgeType is 'directed',  edgeIdx for n1--n2 and edgeIdx for n2--n1 will
%   be different , edgeIdx(n2--n1) = edgeIdx(n1--n2) + G.nEdges
%
% Output:
%   edgeIdxsUnDr,edgeIdxsDr,edgeIdxsRev : Indices of all edge(s) involving the node n
%   (UnDr-undirected and Dr-directed and Rev-reverse edge info)
%
%
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Feb 09,2018. Modified:Feb 09,2018
% To DO : for multiple nodes input, n is an array
Copyright (c) Columbia University Suvrajit Maji 2018 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
'''
def getEdgeIdxsGivenNode(G,n):
    #% EdgeIdx has indices for all edges being treated as distinct (directed)
    #% so edge i-j has a different id say m than edge j-i which is n, regardless
    #% of the type of Adjacency matrix, undirected or directed
    edges = G['EdgeIdx'][n,:].todense() # directed edge info # still has indexing from 1 to nEdges
    edges = edges[edges !=0] # % remove the zero values, note that EdgeIdx had indexing from 1 as matlab so 0 means no edge

    edgeIdxsUnDr = np.array((edges-1) % G['nEdges']).flatten() # % undirected edge indices
    edgeIdxsDr = np.array(edges).flatten() -1# % directed edge indices
    edgeIdxsRev = np.array((edges+G['nEdges']-1)%(2*G['nEdges'])).flatten()# % reverse edge indices

    return (edgeIdxsUnDr,edgeIdxsDr,edgeIdxsRev)



'''
function BPalg = updateBPmessage(BPalg,ePot,msgProd,n,eIdxUnD,eIdxRev,alphaDamp)
Copyright (c) Columbia University Suvrajit Maji 2018 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
'''
def updateBPmessage(BPalg,ePot,msgProd,n,eIdxUnD,eIdxRev,alphaDamp):
    G = BPalg['G']
    #% Compute new message which is the product of edge potential and product of all incoming
    #% messages for a node

    #%UGM max_product function
    if BPalg['options']['maxProduct']:
        #% if 1 then max-product otherwise sum-product belief propagation
        #% max-product (approximates the MAP estimate for tree inference ?)
        new_message_prod = max_product(ePot,msgProd)

    else:
        #% sum-product, marginal probabilities
        new_message_prod = np.matmul(ePot,msgProd)


    enodes =  G['Edges'][eIdxUnD,:]
    nt = enodes[enodes!=n][0] #; % this works because we have only two elements (nodes) in an edge.


    #% use damping factor
    new_message_Damp = (1 - alphaDamp)*BPalg['old_message'][:G['nStates'][nt],eIdxRev] + alphaDamp*new_message_prod

    BPalg['new_message'][0:G['nStates'][nt],eIdxRev] = Normalize(new_message_Damp,1)

    return BPalg



'''
function [nodeBelief,edgeBelief,BPalg] = ComputeBelief(BPalg,nodePot,edgePot)
% Initialise messages with local evidence/node potential
% Input:
%   BPalg: Input BPalg structure
%   nodePot: s x n node potential, s = number of states , n is number of nodes
%   edgePot: s x s x e edge potential,s = number of states , e is number of edges
% Output:
%   nodeBelief: s x n node belief
%   edgeBelief: s x s x e edge belief
%   BPalg: updated BPalg structure with beliefs and messages
%
%
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Feb 02,2018. Modified:Feb 07,2018
Copyright (c) Columbia University Suvrajit Maji 2018
(original matlab version and python debugging)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
'''

def ComputeBelief(BPalg,nodePot,edgePot):

    G = BPalg['G']
    #%edgebelversion = 'mult'; % to do
    edgeBelversion = 'division'

    nodeBelief = np.zeros((G['maxState'],G['nNodes']))

    #% Node belief using the converged/final messages from all neighbors
    prod_of_messages = np.zeros((G['maxState'],G['nNodes']))
    for n in range(G['nNodes']):
        #% neighbors of node n
        [edgeIdxsUnDr, edgeIdxsDr,edgeIdxsRev] = getEdgeIdxsGivenNode(G,n)
        prod_of_messages[:G['nStates'][n],n] = nodePot[:G['nStates'][n],n]*np.prod(BPalg['new_message'][:G['nStates'][n],edgeIdxsDr],axis=1)

        if BPalg['options']['verbose']:
            print('Computing belief for node {}'.format(n))

    #%nodeBelief(1:nStates(n),n) = Normalize(prod_of_messages(1:nStates(n),n),1);
    nodeBelief = Normalize(prod_of_messages,1)

    #% Edge belief given the node beliefs and final messages
    #% division version
    if edgeBelversion=='division':
        edgeBelief = np.zeros((G['nEdges'],G['maxState'],G['maxState']))
        for eij in range(G['nEdges']):
            i = G['Edges'][eij,0]
            j = G['Edges'][eij,1]
            eji = eij+G['nEdges']

            Beli = nodeBelief[:G['nStates'][i],i] / BPalg['new_message'][:G['nStates'][i],eji]
            Belj = nodeBelief[:G['nStates'][j],j] / BPalg['new_message'][:G['nStates'][j],eij]
            edgeBel = np.dot(Beli,Belj.T)*edgePot[eij,:G['nStates'][i],:G['nStates'][j]]

            #% for edge-belief all nSates(i)xnSates(j) terms should sum to 1
            edgeBelief[eij,:G['nStates'][i],:G['nStates'][j]] = Normalize(edgeBel,0) #% normalize the entire state matrix and not just row/column
            if BPalg['options']['verbose']:
                print('Computing belief for edge {}, {}-{}'.format(eij,i,j))
    else:
        edgeBelief = None # TO DO the mult version


    BPalg['nodeBel'] = nodeBelief
    BPalg['edgeBel'] = edgeBelief

    return (nodeBelief,edgeBelief,BPalg)





'''
function BPalg = initializeBPmessage(BPalg,nodePot)
% Initialise messages with local evidence/node potential
% Input:
%   BPalg: Input BP algorithm structure
%   nodePot: s x n node potential, s = number of states, n is number of nodes
% Output:
%   BPalg: initialized BPalg structure with beliefs and messages
%
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Feb 02,2018. Modified:Feb 07,2018

% if all nodes have same number of states

% extract all the fields and parameters to be used
%getFieldsBPalgStruct;
Copyright (c) Columbia University Suvrajit Maji 2018
(original matlab version and python debugging)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)

'''

def initializeBPmessage(BPalg,nodePot):
    G = BPalg['G']

    if (BPalg['options']['eqnStates']):
        #% when all nodes have same number of states 'maxState'
        #% both approaches are equivalent, but for speed purpose we should use this
        #% uniform distribution
        unif_msg = np.ones((G['maxState'],2*G['nEdges']))/G['maxState']
        BPalg['init_message'] = copy.copy(unif_msg) # use b = copy.copy(a) instead of 'a = b'

        if BPalg['options']['verbose']:
            print('Initialized messages from all nodes to their respective neighbors.')
    else:
        #%if variable number of states or would like to initiate different
        #%messages for each node
        BPalg['init_message'] = np.zeros((G['maxState'],2*G['nEdges']))
        nis = G['Edges'][:,0]
        njs = G['Edges'][:,1]
        eij = range(G['nEdges'])
        eji = eij + G['nEdges']

        #% repmat creates variable size column vectors (1:nState(i)) and then rest of the column (nState(i):maxState) is zeros;
        # Mij = arrayfun(@(s) [repmat(1/s,s,1) ; zeros(G.maxState-s,1)],G.nStates(njs),'UniformOutput',false)
        # Mji = arrayfun(@(s) [repmat(1/s,s,1) ; zeros(G.maxState-s,1)],G.nStates(nis),'UniformOutput',false)
        Mij = np.apply_along_axis(lambda x:np.vstack((np.repmat(1./(x+1),(x+1,1)),np.zeros(G['maxState']-x))),1,G['nState'][njs])
        Mji = np.apply_along_axis(lambda x:np.vstack((np.repmat(1./(x+1),(x+1,1)),np.zeros(G['maxState']-x))),1,G['nState'][nis])

        #BPalg.init_message(:,eij) =  cell2mat(Mij) #% from node i to j
        #BPalg.init_message(:,eji) =  cell2mat(Mji) #% from node j to i
        BPalg['init_message'][:,eij] = Mij.tolist()
        BPalg['init_message'][:,eji] = Mji.tolist()

        if BPalg['options']['verbose']:
            print('Initialized messages from all nodes to their respective neighbors.')

    BPalg['new_message'] = copy.copy(unif_msg)  # use b = copy.copy(a) instead of 'a = b'
    BPalg['old_message'] = copy.copy(unif_msg)  # use b = copy.copy(a) instead of 'a = b'

    #% When intialized this is set to zero for the iteration to run more than
    #% once.
    BPalg['convergence'] = 0

    return BPalg



'''
function  BPalg = checkBPconvergence(BPalg,iter)
% Check the convergence of the BP iterations and update
% Input:
%   BPalg: Input BP algorithm structure
%   iter: iteration number
% Output:
%   BPalg: updated BPalg structure with convergence flag and error
%
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Feb 10,2018. Modified:Feb 10,2018
Copyright (c) Columbia University Suvrajit Maji 2018 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
'''

def checkBPconvergence(BPalg):

    BPalg['bpError'] = sum(abs(BPalg['new_message'].flatten()-BPalg['old_message'].flatten()))

    print('Message residual at iter {} = {}'.format(BPalg['bpIter'], BPalg['bpError']))
    if np.isnan(BPalg['bpError']):
        print('Message values contain NaN:Check Node/Edge Potential Values.')
    elif BPalg['options']['verbose']:
        print('Message Propagation residual at iter {} = {}'.format(BPalg['bpIter'], BPalg['bpError']))
    #% we could do the error < tol checking in
    #% the actual bp-loop and not within this function
    if BPalg['bpError'] < BPalg['options']['tol']:
        BPalg['convergence'] = 1
    else:
        BPalg['convergence'] = 0


    if  BPalg['bpIter'] == BPalg['options']['maxIter'] and not BPalg['convergence']:
        print('Warning: Maximum iteration {} reached without convergence: ' \
              'Modify the tolerance and/or increase the maxIter limit and run ' \
              'again'.format(BPalg['options']['maxIter']))

    return BPalg



'''
function [nodeBelief,edgeBelief,BPalg] = GlobalMRFBeliefPropagation(BPalg,nodePot,edgePot)
% Generate the node and edge potential for the Markov Random Field graphical model
% Input:
%   BPalg: Input BPalg structure
%   nodePot: s x n node potential, s = number of states, n is number of nodes
%   edgePot: s x s x e edge potential,s = number of states, e is number of edges
% Output:
%   nodeBelief: s x n node beliefs
%   edgeBelief: s x s x e edge beliefs
%
%
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Feb 02,2018. Modified:Feb 07,2018
%
%
% extract all the fields and parameters to be used
% getFieldsBPalgStruct;
%
Copyright (c) Columbia University Suvrajit Maji 2018 
(original matlab version and python debugging)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    

'''
#[nodeBelief, edgeBelief, BPalg] = GlobalMRFBeliefPropagation.op(BPalg, nodePot, edgePot)
def op(BPalg,nodePot,edgePot):

    G = BPalg['G']
    BPalg = createBPalg(G,BPalg['options'])
    alphaDamp = BPalg['alphaDamp'] # % damping factor
    #tic
    #% initialise messages
    BPalg = initializeBPmessage(BPalg,nodePot)
    BPerrAll = []
    xiter = []

    if BPalg['options']['maxProduct']:
        print('\nNow performing Belief Propagation with max-product ...')
    else:
        print('\nNow performing Belief Propagation with sum-product ...')

    #%%% Belief propagation iterations
    for iter in range(BPalg['options']['maxIter']):
        BPalg['bpIter'] = iter
        print('Belief Propagation Iteration {},'.format(BPalg['bpIter']),)
        #% Each node sends a message to each of its neighbors
        #% the nodes are ordered (default:sequential, 1...nNodes; min. spanning from a single anchor ; multi-anchor )

        graphNodeOrder = BPalg['G']['graphNodeOrder'].flatten('F')
        for n in graphNodeOrder:
            #%Find all neighbors of node n
            #% we need directed edge info from G.EdgeIdx and send a message from node n/i to each of it's neighbor
            [edgeIdxsUnDr,edgeIdxsDr,edgeIdxsRev] = getEdgeIdxsGivenNode(G,n)
            #% edgeIdxsDr(directed),edgeIdxsRev(reverse direction) should always be opposite
	    
            t = 0
            for eij in edgeIdxsUnDr.flatten('F'): # % undirected;
               
                eIdxRev = edgeIdxsRev[t] # %reverse direction;
                #%[i,j] = getNodesGivenEdgeIdx(G,eij); % slower access
                i = G['Edges'][eij,0]
                j = G['Edges'][eij,1] #% G.Edges has undirected edge info, edge i-j and j-i have same node ordering i<j

                if BPalg['options']['verbose']:
                    print('Sending message from node {} to neighbor node {}'.format(i,j))
                    #% edge Potential for edge eij

                ePot_ij = edgePot[eij,:G['nStates'][i],:G['nStates'][j]]
		
                if n == i:
                    ePot_ij = ePot_ij.T

                #% Compute product of all incoming messages to node i except from node j
                incomingMsgProd = nodePot[:G['nStates'][n],n]

                kNbrOfiNOTj = edgeIdxsDr[(edgeIdxsDr != eij) & (edgeIdxsDr != eij+G['nEdges'])] #; % eij is undirected number so eij is always <=G.nEdges
                if len(kNbrOfiNOTj) > 0:
                    incomingMsgProd = incomingMsgProd*(BPalg['new_message'][:G['nStates'][n],kNbrOfiNOTj].prod(axis=1))


                #Compute and update the new message
                BPalg = updateBPmessage(BPalg,ePot_ij,incomingMsgProd,n,eij,eIdxRev,alphaDamp)
                t = t+1

        BPalg = checkBPconvergence(BPalg)
        if np.isnan(BPalg['bpError']):
            break
        if BPalg['convergence'] and np.isinf(BPalg['convergenceIter']):
            BPalg['convergenceIter'] = BPalg['bpIter']
            break

        #% update the old message to the latest message 
        BPalg['old_message'] = copy.copy(BPalg['new_message']) # use b = copy.copy(a) instead of 'a = b'
    
        BPerrAll = np.hstack((BPerrAll,BPalg['bpError']))
        xiter.append(iter)


    #nodeBelief =[]
    #edgeBelief =[]
    if not np.isnan(BPalg['bpError']):
        print('Belief propagation is completed')
    
        if BPalg['convergence']:
            print('\nBelief propagation converged in {} iteration(s)'.format(BPalg['bpIter']+1))
        else:
            print('Belief propagation did not converge after {} iteration(s).The beliefs can be inaccurate.\n'.format(BPalg['bpIter']+1))
        print('Message residual at final iter {} = {}'.format(BPalg['bpIter']+1, BPalg['bpError']))


        print('Computing the beliefs...')
        #% Computing the belief for each node and edge
        nodeBelief,edgeBelief,BPalg = ComputeBelief(BPalg,nodePot,edgePot)

        return (nodeBelief,edgeBelief,BPalg)
    else:
        #_logger.error('NaN error encountered. Check your node and edge potential values.'
        raise ValueError('NaN error encountered. Check your node and edge potential values.')



