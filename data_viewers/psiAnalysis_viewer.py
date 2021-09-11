import os
from pyface.qt import QtGui, QtCore
os.environ['ETS_TOOLKIT'] = 'qt4'
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, Button
import mpl_toolkits.axes_grid1
import matplotlib.path as pltPath
import matplotlib.image as mpimg
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
from pylab import plot, loadtxt, imshow, show, xlabel, ylabel
import pickle
import cv2

#################################################################################
# VIEWER FOR PSI ANALYSIS OUTPUTS
#################################################################################
# HOW TO USE:
## data_viewers folder must be in same parent directory as outputs_{project_name}
## give name of project (below) via 'projName' variable
## load 'ManifoldEM' environment
## change current PD, maxPsis, maxFrames, boxSize as needed
#################################################################################
# Copyright (c) Columbia University Evan Seitz 2020
#################################################################################
# PARAMETERS:
##['IMG1','psirec','tau','psiC1','mu','VX','sdiag','Topo_mean','tauinds']
#################################################################################

projName = 'untitled'
parDir = os.path.abspath('..')
psiDir = os.path.join(parDir, 'outputs_%s/psi_analysis' % (projName))
boxSize = 250 #dimension of each image in stack; e.g. 250x250
maxFrames = 49 #default number per NLSA movie
maxPsis = 2 #total number of psis chosen (1-8)
PD = 0 #current PD
psi = 0 #current psi

if 1: #view NLSA movie frames
    fname = open(os.path.join(psiDir,'S2_prD_%s_psi_%s' % (PD,psi)), 'rb')
    data = pickle.load(fname)

    framesAll = data['IMG1'][0:boxSize**2]

    for i in range(0,maxFrames): #frame index
        frameList = framesAll[:,i]
        frame = np.reshape(frameList, (boxSize,boxSize))

        # plot frames in sequential order (one by one):
        imshow(frame, cmap='gray')
        plt.colorbar()
        plt.show()

if 0: #view Topos
    fname = open(os.path.join(psiDir,'S2_prD_%s_psi_%s' % (PD,psi)), 'rb')
    data = pickle.load(fname)

    toposAll = data['Topo_mean'][0:boxSize**2] #IMG1

    toposList = toposAll[:,1]
    topos = np.reshape(toposList, (boxSize,boxSize))

    plt.title('topos %s' % (i+1))
    imshow(topos, cmap='gray')
    plt.colorbar()
    plt.show()
