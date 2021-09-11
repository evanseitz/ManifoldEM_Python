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

#################################################################################
# DIFFUSION MAPS VIEWER
#################################################################################
# HOW TO USE:
## data_viewers folder must be in same parent directory as outputs_{project_name}
## give name of project (below) via 'projName' variable
## load 'ManifoldEM' environment
## change current PD, dimRows, dimCols as needed
#################################################################################
# Copyright (c) Columbia University Evan Seitz 2020
#################################################################################

projName = 'untitled'
parDir = os.path.abspath('..')
dmDir = os.path.join(parDir, 'outputs_%s/diff_maps' % (projName))
PD = 0 #current PD (zero indexing)

if 1: #plot of all eigenfunctions v1 (up to 'dimRows/Cols') vs all others (v1+i)
    fname = open(os.path.join(dmDir,'gC_trimmed_psi_prD_%s' % (PD)), 'rb')
    data = pickle.load(fname)
    DM = data['psi']
    print(np.shape(DM))
    m = np.shape(DM)[0]
    enum = np.arange(1,m+1)
    
    fig = plt.figure() 
    dimRows = 4 #6
    dimCols = 5 #8
    s = 20
    lw = .5
    idx = 1
    cmap = 'gist_rainbow'
    for v1 in range(1,dimRows+1):
        for v2 in range(v1, v1+dimCols):
            plt.subplot(dimRows, dimCols, idx)
            plt.scatter(DM[:,v1-1], DM[:,v2-1], c=enum, cmap=cmap, s=s, linewidths=lw, edgecolor='k') #gist_rainbow, nipy_spectral
            plt.xlabel(r'$\Psi_{%s}$' % (v1-1), fontsize=12, labelpad=5)
            plt.ylabel(r'$\Psi_{%s}$' % (v2-1), fontsize=12, labelpad=2.5)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
            plt.rc('font', size=6)

            if 1:
                frame = plt.gca()
                frame.axes.xaxis.set_ticklabels([])
                frame.axes.yaxis.set_ticklabels([])
                plt.gca().set_xticks([])
                plt.xticks([])
                plt.gca().set_yticks([])
                plt.yticks([])
            else:
                plt.tick_params(axis="x", labelsize=6)
                plt.tick_params(axis="y", labelsize=6) 
            idx += 1 
    plt.tight_layout()
    plt.show()
