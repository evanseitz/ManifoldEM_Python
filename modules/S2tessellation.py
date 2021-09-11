import distribute3Sphere
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import logging, sys
from sklearn.neighbors import NearestNeighbors
#from scipy.spatial import Delaunay

'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def conjugate_bin(S20,S2,CG1,NIND):
   # compute affinities among the bins
   nbins = S20.shape[1]
   npart = int(S2.shape[1] / 2)
   mat = np.zeros((nbins, npart), dtype=int)
   for i in NIND:
       # filling in the occupied indices modulo npart
       # i.e., only the original part indices
       mat[i, CG1[i] % npart] = 1
   # affinity calculation
   aff = np.matmul(mat, mat.T)
   arank = np.argsort(aff)
   cbin = arank[:,nbins-2:]
   for i in NIND:
      if cbin[i,0] == i:
         cbin[i,0] = cbin[i,1]
   cbin = cbin[:,0]
   return cbin

def get_S2(q):
   try:
      assert(q.shape[0] > 3)
   except AssertionError:
      _logger.error('subroutine get_S2: q has wrong dimensions')
      _logger.exception('subroutine get_S2: q has wrong diemnsions')
      sys.exit(1)
      raise
      # projection angles   
   S2 = 2*np.vstack((q[1,:]*q[3,:]-q[0,:]*q[2,:],\
                     q[0,:]*q[1,:]+q[2,:]*q[3,:], \
                     q[0,:]**2 + q[3,:]**2-0.5))                  
   """
   From the original matlab code: 
   S2 = 2*[q(2,:).*q(4,:)-q(1,:).*q(3,:);
           q(1,:).*q(2,:)+q(3,:).*q(4,:);
           q(1,:).^2+q(4,:).^2-0.5];
   """
   return S2

def classS2(X,Q):
   nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
   distances, IND = nbrs.kneighbors(Q)
   NC = np.bincount(IND[:,0].squeeze())
   return (IND,NC)

def op(q,shAngWidth,PDsizeTh,visual,thres,*fig):

   nG = np.floor(4*np.pi / (shAngWidth**2)).astype(int)
   # reference angles
   S20,it = distribute3Sphere.op(nG)
   #print S20.shape
   S20 = S20.T
   #nS = q.shape[1]
   # projection angles
   S2 = get_S2(q)

   IND, NC = classS2(S20.T, S2.T)

   # non-thresholded
   #NIND = (NC >= 0).nonzero()[0] # NIND is just [0:S20.shape[1]]
   CG1 = []
   for i in range(S20.shape[1]):
      a = (IND == i).nonzero()[0]
      # upper-thresholded
      #if len(a) > thres:
      #   a = a[:thres]
      CG1.append(a)

   CG1 = np.array(CG1, dtype=object) #added: , dtype=object)
   #print('CG1', CG1)

   # lower-thresholded
   mid = np.floor(S20.shape[1]/2).astype(int)
   # halving first
   NC1 = NC[:mid]
   NC2 = NC[mid:]

   '''
   NIND = (NC1 >= PDsizeTh).nonzero()[0]
   '''
   NIND = []
   if len(NC1) >= len(NC2):
      pd = 0 #PD index
      for i in NC1: #NC1 is the occupancy of PrD=pd
         if i >= PDsizeTh:
            NIND.append(pd)
         pd += 1
   else:
      pd = mid #PD index
      for i in NC2: #NC2 is the occupancy of PrD=pd
         if i >= PDsizeTh:
            NIND.append(pd)
         pd += 1

   '''
   if len(NC1) < len(NC2):
       NIND = (NC2 >= PDsizeTh).nonzero()[0]
       '''

   # find the "conjugate" bins
   #cbin = conjugate_bin(S20, S2, CG1, NIND)
   #NIND1 = np.hstack((NIND,cbin[NIND]))
   #NIND1 = np.unique(NIND1)
   #print 'NINd1=',NIND1
   S20_th = S20[:,NIND]

   CG = []
   for i in range(len(NIND)):
      a = (IND == NIND[i]).nonzero()[0]
      # upper-thresholded
      if len(a) > thres:
         a = a[:thres]
      CG.append(a)

      #print 'ind=',i
      #print 'list of particles=',a
      #print 'shape =',a.shape
      #print ''

   '''
   for i in NIND:
      print(len(CG1[i])) #sanity-check
      '''

   if visual:
      # fig = plt.figure(figsize=(4,4))
      ax = Axes3D(fig)
      ax.set_xlim3d(-1, 1)
      ax.set_ylim3d(-1, 1)
      ax.set_zlim3d(-1, 1)

      for i in range(np.floor(CG.shape[1]) / 2):
         a = CG[i]
         ax.scatter(S2(1, a[::40]), S2(1, a[::40]), S2(2, a[::40]),marker='o',s=20,alpha=0.6)

   return (CG1, CG, nG, S2, S20_th, S20, NC)
   #CG1: list of lists, each of which is a list of image indices within one PD
   #CG: thresholded version of CG1
   #nG: approximate number of tessellated bins
   #S2: cartesian coordinates of each of particles' angular position on S2 sphere
   #S20_th: thresholded version of S20
   #S20: cartesian coordinates of each bin-center on S2 sphere
   #NC: list of occupancies of each PD



#if __name__ == '__main__':
#   op(ang,df,Shan_width,visual,GCnum,fig,flip=True)
