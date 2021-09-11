import logging, sys
import numpy as np
from scipy import optimize
import qMult_bsx

#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

'''Converts a quaternion to corresponding rotation sequence in Spider 3D convention
Due to the 3D convention for all angles there is no need to negate psi
q: 4x1
Implementation by optimization
Copyright(c) UWM, Peter Schwander Jan. 31, 2014, Mar. 19, 2014 matlab version
version = 'q2Spider, V1.1'

Copyright (c) Columbia University Hstau Liao 2018 (python version)    
'''

def op(q):

    # assert unit quaternion
    q = q / np.sqrt(sum(q**2))

    def dev1(a):
        q1 = np.array([np.cos(a[0] / 2.), 0., 0., -np.sin(a[0] / 2.)])  # see write-up
        q2 = np.array([np.cos(a[1] / 2.), 0, -np.sin(a[1] / 2.), 0.])
        q3 = np.array([np.cos(a[2] / 2.), 0., 0., -np.sin(a[2] / 2)])
        F = q - qMult_bsx.op(q3, qMult_bsx.op(q2,q1)).flatten()
        #print a,F
        return F

    lb = -np.inf
    ub = np.inf
    # options = optimset('Display', 'iter', 'TolFun', 1e-16);
    tol = 1e-12
    exitflag = np.nan
    resnorm = np.inf

    a0 = np.array([0,0,0])
    nTry = 0
    # tic
    res = optimize.least_squares(dev1, a0, bounds=(lb, ub), ftol=tol)
    a = res.x
    '''
    while exitflag != 1 or resnorm > 1e-12:
        res = optimize.least_squares(dev1,a0,bounds=(lb, ub), ftol=tol)
        a=res.x
        resnorm = res.cost
        exitflag = res.status
        print resnorm, exitflag
        nTry = nTry + 1
        a0 = (np.pi / 2) * (np.random.uniform(0,1,3) - 0.5) # use random guess for next try end
    # toc
    '''
    phi = a[0]  #% (2*np.pi) * 180 / np.pi
    theta = a[1]#% (2*np.pi) * 180 / np.pi
    psi = a[2]  #% (2*np.pi) * 180 / np.pi

    #% nTry
    return (phi, theta, psi)

if __name__ == '__main__':
   op()

