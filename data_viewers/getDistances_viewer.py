import os, sys
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

#################################################################################
# VIEWER FOR DISTANCE OUTPUTS
#################################################################################
# HOW TO USE:
## data_viewers folder must be in same parent directory as outputs_{project_name}
## give name of project (below) via 'projName' variable
## load 'ManifoldEM' environment
## change current PD via `PD` variable and select parameter to view below
#################################################################################
# Copyright (c) Columbia University Evan Seitz 2020
#################################################################################
# PARAMETERS:
## ['D','ind','q','df','CTF','imgAll','PD','PDs','Psis','imgAvg','imgAvgFlip',
## 'imgAllFlip','imgLabels','Dnom','Nom','imgAllIntensity','version','options']
#################################################################################

projName = 'untitled'
parDir = os.path.abspath('..')
outDir = os.path.join(parDir, 'outputs_%s/distances' % projName)

PD = 0 #PD index (from index 0 to N-1)

if 1: #imgAvg: Image Average
    #fname = open(os.path.join(outDir, 'IMGs_prD_%s' % PD), 'r')
    fname = open(os.path.join(outDir,'IMGs_prD_%s' % (PD)))
    data = pickle.load(fname)
    img = data['imgAvg']
    imshow(img, cmap='gray')
    plt.title('PD %s Image Average' % (PD+1))
    plt.colorbar()
    plt.show()

if 0: #D: Distance Matrix
    fname = open(os.path.join(outDir, 'IMGs_prD_%s' % PD), 'rb')
    data = pickle.load(fname)
    img = data['D']
    imshow(img, cmap='jet')
    plt.title('PD %s Distance Matrix' % (PD+1))
    plt.colorbar()
    plt.show()
