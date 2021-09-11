import numpy as np
import get_fit_1D_open_manifold_3D_param
import solve_d_R_d_tau_p_3D
import a
import linalg
#from scipy.io import loadmat

'''
function [a,b,tau] = fit_1D_open_manifold_3D(psi)
% 
% fit_1D_open_manifold_3D
% 
% fit the eigenvectors for a 1D open manifold to the model
% x_ij = a_j cos(j*pi*tau_i) + b_j.
% 
% j goes from 1 to 3 (this is only for 3D systems).
% 
% i goes from 1 to nS where nS is the number of data points to be fitted.
% 
% For a fixed set of a_j and b_j, j=1:3, tau_i for i=1:nS are
% obtained by putting dR/d(tau_i) to zero.
% 
% For a fixed set of tau_i, i=1:nS, a_j and b_j for j=1:3 are
% obtained by solving 3 sets of 2x2 linear equations.
% 
% Fit parameters and initial set of {\tau} are specified in
% 
%   get_fit_1D_open_manifold_3D_param.m
% 
% copyright (c) Russell Fung 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    

  global p nDim a b x x_fit

  '''
'''
def plot_fitted_curve(hFig):
    global x x_fit
    h = plt.figure(hFig)
    hsp = plt.subplot(2,2,1)
    plot3(x(:,1),x(:,2),x(:,3),'b.','lineWidth',1);
  hold on
  plot3(x_fit(:,1),x_fit(:,2),x_fit(:,3),'g.','lineWidth',1);
  hold off
  set(hsp,'lineWidth',2,'fontSize',15);
  hsp = subplot(2,2,2);
  plotRF(hsp,x(:,1),x(:,2),'','','','b.');
  addplotRF(hsp,x_fit(:,1),x_fit(:,2),'g.');
  hsp = subplot(2,2,3);
  plotRF(hsp,x(:,1),x(:,3),'','','','b.');
  addplotRF(hsp,x_fit(:,1),x_fit(:,3),'g.');
  hsp = subplot(2,2,4);
  plotRF(hsp,x(:,2),x(:,3),'','','','b.');
  addplotRF(hsp,x_fit(:,2),x_fit(:,3),'g.');
  drawnow
%end
'''
eps = 1e-4


#global maxIter,delta_a_max, delta_b_max,delta_tau_max,a_b_tau_result

def op(psi):
    a.init()
    #global p, nDim, a, b, x, x_fit
    a.nDim = 3
    #tau = get_fit_1D_open_manifold_3D_param

    tau = get_fit_1D_open_manifold_3D_param.op(psi)

    aux = np.zeros((tau.shape[0],5))   #added
    nS = a.x.shape[0]
  
    for iter in range(1,a.maxIter+1):
        #string ='iteration ' + str(iter)
        #print string
        '''
        #%%%%%%%%%%%%%%%%%%%%%
        #% solve for a and b %
        #%%%%%%%%%%%%%%%%%%%%%
        '''
        a_old = a.a
        b_old = a.b
        j_pi_tau = np.dot(tau,np.pi*np.array([[1,2,3]]))
        cos_j_pi_tau = np.cos(j_pi_tau)
        A11 = np.sum(cos_j_pi_tau**2, axis=0)
        A12 = np.sum(cos_j_pi_tau, axis=0)
        A21 = A12
        A22 = nS
        x_cos_j_pi_tau = a.x*cos_j_pi_tau
        b1 = np.sum(x_cos_j_pi_tau, axis=0)
        b2 = np.sum(a.x, axis=0)
        coeff = np.zeros((2,3))
        for qq in range(3):
            A = np.array([[A11[qq],A12[qq]],[A21[qq], A22]])
            b = np.array([b1[qq], b2[qq]])
            #coeff[:,qq] = linalg.op(A,b)
            coeff[:,qq] = np.linalg.lstsq(A, b)[0]
        

        a.a = coeff[0,:]
        a.b = coeff[1,:]
        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%
        #% plot the fitted curve %
        %%%%%%%%%%%%%%%%%%%%%%%%%
        '''
        j_pi_tau = np.dot(np.linspace(0,1,1000).reshape(-1,1),np.array([[1,2,3]]))*np.pi
        cos_j_pi_tau = np.cos(j_pi_tau)
        tmp = a.a*cos_j_pi_tau
        a.x_fit = tmp + a.b
        #%plot_fitted_curve(iter)
        '''
        %%%%%%%%%%%%%%%%%
        #% solve for tau %
        %%%%%%%%%%%%%%%%%
        '''
        tau_old = tau
        for a.p in range(nS):
          tau[a.p],beta = solve_d_R_d_tau_p_3D.op()   #added
          for kk in range(beta.shape[0]):
              aux[a.p,kk] = beta[kk]
        '''
        if iter == 0:
            data = loadmat('aux0.mat')  # (this is for < v7.3
        elif iter == 1:
            data = loadmat('aux1.mat')  # (this is for < v7.3
        else:
            data = loadmat('aux2.mat')  # (this is for < v7.3
        imaux = data['aux']
        plt.subplot(2, 2, 1)
        plt.imshow(aux, cmap=plt.get_cmap('gray'),aspect=0.1)
        plt.title('aux')
        plt.subplot(2, 2, 2)
        plt.imshow(imaux, cmap=plt.get_cmap('gray'), aspect=0.1)
        plt.title('imaux')
        plt.show()
        '''

        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #% calculate the changes in fitting parameters %
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        '''
        delta_a = np.fabs(a.a-a_old)/(np.fabs(a.a)+eps)
        delta_b = np.fabs(a.b-b_old)/(np.fabs(a.b)+eps)
        delta_tau = np.fabs(tau-tau_old)
        delta_a = max(delta_a)*100
        delta_b = max(delta_b)*100
        delta_tau = max(delta_tau)
        #print '  changes in fitting parameters: \n'
        #string = '  amplitudes: '+ str(delta_a) + '\n' + \
        #         '  offsets: ' + str(delta_b) + ' \n' +\
        #         '  values of tau: ' + str(delta_tau) + ' \n'
        #print string
        if (delta_a<a.delta_a_max) and (delta_b < a.delta_b_max) and (delta_tau < a.delta_tau_max):
            break

    return (a.a,a.b,tau)
