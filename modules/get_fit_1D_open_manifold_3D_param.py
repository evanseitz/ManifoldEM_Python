import numpy as np
import solve_d_R_d_tau_p_3D
import a
import linalg

'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
'''

def op(psi):
    #psi = -psi
    #global psi, x, a, b
    #global maxIter,delta_a_max, delta_b_max,delta_tau_max,a_b_tau_result

    a.maxIter = 100       # % maximum number of iterations, each iteration determines
                        #%   optimum sets of {a,b} and {\tau} in turns
    a.delta_a_max = 1     # % maximum percentage change in amplitudes
    a.delta_b_max = 1      #% maximum percentage change in offsets
    a.delta_tau_max = 0.01 #% maximum change in values of tau
    a.a_b_tau_result = 'a_b_tau_result.pkl' #% save final results here
  
    nS = psi.shape[0]
    a.x = psi[:,0:3]
    #% (1) initial guesses for {a,b} obtained by fitting data in 2D
    #% (2) initial guesses for {tau} obtained by setting d(R)/d(tau) to zero
  
    X = psi[:,0]
    Z = psi[:,2]
    X2 = X*X
    X3 = X2*X
    X4 = X2*X2
    X5 = X3*X2
    X6 = X3*X3
    sumX = np.sum(X)
    sumX2 = np.sum(X2)
    sumX3 = np.sum(X3)
    sumX4 = np.sum(X4)
    sumX5 = np.sum(X5)
    sumX6 = np.sum(X6)
    sumZ = np.sum(Z)
    sumXZ = np.dot(X.T,Z)
    sumX2Z = np.dot(X2.T,Z)
    sumX3Z = np.dot(X3.T,Z)
    A = np.array([[sumX6, sumX5, sumX4, sumX3],
                  [sumX5, sumX4, sumX3, sumX2],
                  [sumX4, sumX3, sumX2, sumX],
                  [sumX3, sumX2, sumX,  nS   ]])
    b = np.array([sumX3Z, sumX2Z, sumXZ, sumZ])
    #coeff = linalg.op(A, b)
    coeff = np.linalg.lstsq(A, b)[0]
    D = coeff[0]
    E = coeff[1]
    F = coeff[2]
    G = coeff[3]
    disc = E*E - 3*D*F
    if disc < 0:
        #print 'disc=',disc
        disc = 0.
    if np.absolute(D) < 1e-8:
        #print 'D=',D
        D = 1e-8
    a1 = (2.*np.sqrt(disc))/(3.*D)
    a3 = (2.*disc**(3/2.))/(27.*D*D)
    b1 = -E/(3*D)
    b3 = (2.*E*E*E)/(27.*D*D)-(E*F)/(3*D) + G
  
    Xb = X-2*b1
    Y = psi[:,1]
    XXb = X*Xb
    X2Xb2 = XXb*XXb
    sumXXb = np.sum(XXb)
    sumX2Xb2 = np.sum(X2Xb2)
    sumY = np.sum(Y)
    sumXXbY = np.dot(XXb.T,Y)
    A = np.array([[sumX2Xb2, sumXXb],[sumXXb, nS]])
    b = np.array([sumXXbY, sumY])
    coeff = np.linalg.lstsq(A, b)[0]
    #coeff = linalg.op(A,b)
    A = coeff[0]
    C = coeff[1]
    a2 = 2.*A*disc/(9.*D*D)
    b2 = C+(A*E*E)/(9.*D*D)-(2.*A*F)/(3.*D)
    a.a = np.array([a1, a2, a3])
    a.b = np.array([b1, b2, b3])


    tau = np.zeros((nS,1))

    for a.p in range(nS):
        tau[a.p],beta = solve_d_R_d_tau_p_3D.op()  #added

    return tau

