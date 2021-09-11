import numpy as np
from scipy.ndimage import affine_transform,map_coordinates,rotate
#from skimage.transform import warp
#from skimage.transform import ProjectiveTransform
#from scipy.spatial.transform import Rotation
from transformations import  euler_from_quaternion,quaternion_from_euler,quaternion_conjugate
import math
from math import cos,sin
import sys
sys.path.append('../')
import q2Spider
import mrcfile
import myio, p
from transformations import  euler_from_quaternion,quaternion_from_euler,quaternion_matrix

'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Sonya Hanson 2018 (python version)
Copyright (c) Columbia University Shayan S 2018 (python version)
'''

'''
% function rhoSym = symmetrize(rho, sym, center)
%
% This function returns symmetrizes a 3D density
%
% rho (DxDxD)   3-dim density
% sym (Nx3)     List of N rotational symmetry elements, Euler ZYZ
% center (1x3)  Symmetry center
%
% Programmed April 2013, Peter Schwander
% Copyright (c) UWM, 2013 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Sonya Hanson 2018 (python version)
Copyright (c) Columbia University Shayan S 2018 (python version)
'''

#New code for projectMask: Suvrajit Maji
#Created March 2019, Modified: June 2019

#Euler angles to rotation matrix
def eulerRotMatrix3DSpider(Phi, Theta, Psi,deg):

    if deg:
        Phi =  math.radians(Phi)
        Theta =  math.radians(Theta)
        Psi =  math.radians(Psi)

    R = np.array([[cos(Phi)*cos(Psi)*cos(Theta)+(-1)*sin(Phi)*sin(Psi), cos(Psi)*cos(Theta)*sin(Phi)+cos(Phi)*sin(Psi),(-1)*cos(Psi)*sin(Theta)],
                 [(-1)*cos(Psi)*sin(Phi)+(-1)*cos(Phi)*cos(Theta)*sin(Psi),cos(Phi)*cos(Psi)+(-1)*cos(Theta)*sin(Phi)*sin(Psi),sin(Psi)*sin(Theta)],
                 [cos(Phi)*sin(Theta),sin(Phi)*sin(Theta),cos(Theta)]
                 ])

    return R

## this usage of affine transform function with separate rotation and translation (offset)
def rotateVolumeEuler(vol,sym,deg):
    dims = vol.shape
    rotmat = eulerRotMatrix3DSpider(sym[2], sym[1], sym[0],deg)

    # if input euler angles are not already negative, then we have to take the inverse.
    #T_inv = np.linalg.inv(rotmat)
    T_inv = rotmat
    #print 'Euler-rotmat',rotmat

    c_in =  0.5*np.array(dims)
    c_out = 0.5*np.array(dims)
    cen_offset = c_in - np.dot(T_inv, c_out)
    #print 'cen_offset', cen_offset
    rho = affine_transform(input=vol,matrix=T_inv,offset=cen_offset,output_shape=dims,mode='nearest')
    #print 'rho',rho
    return rho

## alternate way using quaternion to rotation matrix
## not used currently
def rotateVolumeQuat(vol,q):
    dims = vol.shape
    rotmat = quaternion_matrix(q).T
    # if input euler angles are not already negative for inverse transform
    rotmat= np.linalg.inv(rotmat)
    T_inv = rotmat[0:3,0:3]
    #print 'Quat-rotmat',rotmat

    c_in =  0.5*np.array(dims)
    c_out = 0.5*np.array(dims)
    cen_offset= c_in - np.dot(T_inv, c_out)
    rho = affine_transform(input=vol,matrix=T_inv,offset=cen_offset,output_shape=dims,mode='nearest')
    return rho


# get the euler angles from PD
def getEuler_from_PD(PD,deg):
    Qr = np.array([1 + PD[2], PD[1], -PD[0], 0]).T
    q1 = Qr/np.sqrt(sum(Qr**2))
    #print 'q1',q1
    if not deg:
        phi, theta, psi = q2Spider.op(q1)
    elif deg==1:
        #phi, theta, psi = q2Spider_mod.op(q1,deg)
        phi, theta, psi = q2Spider.op(q1)
        phi = phi * 180 / np.pi
        theta = theta * 180 / np.pi
        psi = psi * 180 / np.pi
    elif deg==2:
        phi, theta, psi = q2Spider.op(q1)
        phi = (phi % (2*np.pi)) * 180 / np.pi
        theta = (theta % (2*np.pi)) * 180 / np.pi
        psi = (psi % (2*np.pi)) * 180 / np.pi

    sym = np.array([phi,theta,psi])

    return sym,q1


def op(vol,PD):

    vol =  np.swapaxes(vol,0,2)
    nPix = vol.shape[0]
    deg = 0

    sym,q = getEuler_from_PD(PD,deg)
    sym[2] = 0 # as psi=0 , the input images have already been inplane rotated
    sym = sym*(-1.) # for inverse transformation

    #print 'sym-angles-euler-PD',sym*180.0/np.pi
    rho = rotateVolumeEuler(vol,sym,deg)

    msk = np.sum(rho,axis=2) # axis= 2 is z slice after swapping axes(0,2)
    msk = msk.reshape(nPix,nPix).T
    #print 'mask pix',np.sum(msk>0)
    msk = msk>1
    return msk

if __name__ == '__main__':
    vol = np.ones((10,10,10))
    PD = np.array([0.3,0.4,0.5])
    op(vol,PD)
