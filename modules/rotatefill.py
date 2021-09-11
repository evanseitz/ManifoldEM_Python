'''function [out] = imrotateFill(inp, angle)
% function  [out] = imrotateFill(inp)
% Rotates an 2D image couterclockwise by angle in degrees
% Output image has the same dimension as input.
% Undefined regions are filled in by repeating the original image
% Note: input images must be square
%
% Copyright (c) UWM, Peter Schwander Mar. 20, 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
version = 'imrotateFill, V0.9';

Ported to python. Hstau Liao Oct. 2016
'''

import numpy as np
import logging,sys
import math
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt

def op(input, angle, visual=False):
    nPix = input.shape[0]
    inpRep = np.tile(input, (3, 3))
    outRep = rotate(inpRep, angle, reshape=False)
    out = outRep[nPix:2 * nPix, nPix:2 * nPix]

    if visual:
        plt.subplot(2, 2, 1)
        plt.imshow(input,cmap = plt.get_cmap('gray'))
        plt.title('Input')
        plt.subplot(2, 2, 2)
        plt.imshow(out, cmap=plt.get_cmap('gray'))
        plt.title('Output')
        plt.subplot(2, 2, 3)
        plt.imshow(inpRep, cmap=plt.get_cmap('gray'))
        plt.title('Input 3x3')
        plt.subplot(2, 2, 4)
        plt.imshow(outRep, cmap=plt.get_cmap('gray'))
        plt.title('Output 3x3')
        plt.show()
    return out

if __name__ == '__main__':

    # tested using a 6x6 image
    img = np.loadtxt(sys.argv[1])
    ang = float(sys.argv[2]) # in degrees
    visual = bool(sys.argv[3])
    result = op(img,ang,visual)


