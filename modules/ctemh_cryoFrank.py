"""function y = ctemh_cryoFrank(k,params)
% from Kirkland, adapted for cryo (EMAN1) by P. Schwander
% Version V 1.1
% Copyright (c) UWM, Peter Schwander 2010 MATLAB version
% '''
% Copyright (c) Columbia University Hstau Liao 2018 (python version)    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Here, the damping envelope is characterized by a single parameter B 
% see J. Frank
% params(1)   Cs in mm
% params(2)   df in Angstrom, a positive value is underfocus
% params(3)   Electron energy in keV
% params(4)   B in A-2

% Note: we assume |k| = s

"""
import sys
import numpy as np
import math

def op(k,params):
   Cs = params[0]*1.0e7
   df = params[1]
   kev = params[2]
   B = params[3]
   ampc = params[4]
   mo = 511.0
   hc = 12.3986
   wav = (2*mo)+kev
   wav = hc/np.sqrt(wav*kev)
   w1 = np.pi*Cs*wav*wav*wav
   w2 = np.pi*wav*df
   k2 = k*k
   #wi = exp(-2*B*k2); % B. Sander et al. / Journal of Structural Biology 142 (2003) 392?401, CHECKCHECK
   sigm = B/math.sqrt(2*math.log(2)) # B is Gaussian Env. Halfwidth
   #sigm = B/2;
   wi = np.exp(-k2/(2*sigm**2))
   wr = (0.5*w1*k2-w2)*k2 # gam = (pi/2)Cs lam^3 k^4 - pi lam df k^2 
   
   y = (np.sin(wr)-ampc*np.cos(wr))*wi
   return y

if __name__ == '__main__':

    k = sys.argv[1]
    params = sys.argv[2]
    result = op(k,params)
    
      


