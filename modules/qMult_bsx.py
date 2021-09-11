"""import globfunction p = qMult_bsx(q,s)
for any number of quaternions N
q is 4xN or 4x1
s is 4xN or 4x1
"""

'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
'''
import sys
import numpy as np

def op(q,s):
   # if 1-dim vector
   if len(q.shape) < 2:
      q = q.reshape(-1,1)
   if len(s.shape) < 2:
      s = s.reshape(-1,1)
   try: 
      assert (q.shape[0] > 3 and s.shape[0] > 3)
   except AssertionError:
      print('subroutine qMult_bsx: some vector have less than 4 elements') 
   q0 = q[0,:]
   qv = q[1:4,:]
   s0 = s[0,:]
   sv = s[1:4,:]

   c = np.vstack((qv[1,:]*sv[2,:] - qv[2,:]*sv[1,:],
                 qv[2,:]*sv[0,:] - qv[0,:]*sv[2,:],
                  qv[0,:]*sv[1,:] - qv[1,:]*sv[0,:]))

   p = np.vstack((q0*s0-np.sum(qv*sv,axis=0),q0*sv+s0*qv+c))
   return p

"""c = [bsxfun(@times,qv(2,:),sv(3,:))-bsxfun(@times,qv(3,:),sv(2,:));
        bsxfun(@times,qv(3,:),sv(1,:))-bsxfun(@times,qv(1,:),sv(3,:));
        bsxfun(@times,qv(1,:),sv(2,:))-bsxfun(@times,qv(2,:),sv(1,:))];
p = [bsxfun(@times,q0,s0)-sum(bsxfun(@times,qv,sv));bsxfun(@times,q0,sv)+bsxfun(@times,s0,qv)+c];
"""
if __name__ == '__main__':

    q = sys.argv[1]
    s = sys.argv[2]
    result = op(q,s)
    #print numpy.array(table)
      


