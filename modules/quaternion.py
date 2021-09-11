import numpy as np
from scipy import optimize

'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
'''

def qMult_bsx(q,s):
    """import globfunction p = qMult_bsx(q,s)
    for any number of quaternions N
    q is 4xN or 4x1
    s is 4xN or 4x1
    """
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

def q2Spider(q):
    '''Converts a quaternion to corresponding rotation sequence in Spider 3D convention
    Due to the 3D convention for all angles there is no need to negate psi
    q: 4x1
    Implementation by optimization
    Copyright(c) UWM, Peter Schwander Jan. 31, 2014, Mar. 19, 2014 matlab version
    version = 'q2Spider, V1.1'

    Copyright (c) Columbia University Hstau Liao 2018 (python version)
    '''
    # assert unit quaternion
    q = q / np.sqrt(sum(q**2))

    def dev1(a):
        q1 = np.array([np.cos(a[0] / 2.), 0., 0., -np.sin(a[0] / 2.)])  # see write-up
        q2 = np.array([np.cos(a[1] / 2.), 0, -np.sin(a[1] / 2.), 0.])
        q3 = np.array([np.cos(a[2] / 2.), 0., 0., -np.sin(a[2] / 2)])
        F = q - qMult_bsx(q3, qMult_bsx(q2,q1)).flatten()
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


def psi_ang(PD):
    Qr = np.array([1 + PD[2], PD[1], -PD[0], 0])
    Qr = Qr / np.sqrt(np.sum(Qr**2))
    phi,theta,psi = q2Spider(Qr)

    phi = np.mod(phi, 2 * np.pi) * (180 / np.pi)
    theta = np.mod(theta, 2 * np.pi) * (180 / np.pi)
    psi = 0.0 # already done in getDistances np.mod(psi,2*np.pi)*(180/np.pi)
    return (phi,theta,psi)

def calc_avg_pd(q,nS):
    # Calculate average projection directions (from matlab code)
    """PDs = 2*[q(2,:).*q(4,:) - q(1,:).*q(3,:);...
    q(1,:).*q(2,:) + q(3,:).*q(4,:); ...
    q(1,:).^2 + q(4,:).^2 - ones(1,nS)/2 ];
    """
    PDs = 2*np.vstack((q[1,:]*q[3,:]-q[0,:]*q[2,:],
                 q[0,:]*q[1,:] + q[2,:]*q[3,:],
		 q[0,:]**2 + q[3,:]**2 - np.ones((1,nS))/2.0))

    return PDs
