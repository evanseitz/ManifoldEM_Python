import numpy as np

'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
'''

def tidyUp(D,EV):
    order = np.argsort(D)[::-1]
    D = np.sort(D)[::-1]
    EV = EV[:,order]
    sqrtD = np.sqrt(D)
    S = np.diag(sqrtD)
    invS = np.diag(1./sqrtD)
    return (D,EV,S,invS)

def op(A):
    D1,D2 = A.shape
    #print D1,D2
    if D1 > D2:
        tt = np.matmul(A.T,A)
        #print 'tt is', tt[0:4,0:4]
        D,V = np.linalg.eigh(np.matmul(A.T,A))
        #print 'D is', D[0:8]
        #print 'Vis is', V[0:8, 0:8]
        D,V,S,invS = tidyUp(D,V)
        U = np.matmul(A,np.matmul(V,invS))
    else:
        D,U = np.linalg.eigh(np.matmul(A,A.T))
        D,U,S,invS = tidyUp(D,U)
        V = np.matmul(A.T,np.matmul(U,invS))

    #% end function svdR

    return (U,S,V)