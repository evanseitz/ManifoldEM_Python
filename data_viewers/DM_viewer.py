import os
import shutil
import matplotlib
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
from matplotlib import cm


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

if 1: #render with LaTeX font for figures
    rc('text', usetex=True)
    rc('font', family='serif')
    
# user parameters:
projName = 'untitled'
PD = 1 #current PD (zero indexing)
Batch = True #if True, batch print all PD embeddings to files in current directory

# don't change:
pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
parDir = os.path.abspath('..')
dmDir = os.path.join(parDir, 'outputs_%s/diff_maps' % (projName))

if Batch is False: #standard use (view one PD embedding at a time)
    if 1: #plot of all eigenfunctions v1 (up to 'dimRows/Cols') vs all others (v1+i)
        fname = open(os.path.join(dmDir,'gC_trimmed_psi_prD_%s' % (PD)), 'rb')
        data = pickle.load(fname)
        DM = data['psi']
        
        s = 20
        lw = .5
        idx = 1
        print(np.shape(DM))
        m = np.shape(DM)[0]
        enum = np.arange(1,m+1)
        fig = plt.figure() 
        dimRows = 6 #4
        dimCols = 8 #5
    
        for v1 in range(1,dimRows+1):
            for v2 in range(v1, v1+dimCols):
                plt.subplot(dimRows, dimCols, idx)
                plt.scatter(DM[:,v1-1], DM[:,v2-1], s=s, edgecolor='k', linewidths=.1, zorder=1)
                plt.xlabel(r'$\Psi_{%s}$' % (v1), fontsize=12, labelpad=5)
                plt.ylabel(r'$\Psi_{%s}$' % (v2), fontsize=12, labelpad=2.5)
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
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97, wspace=0.25, hspace=0.25)
        plt.show()
        
    if 0: #3d diffusion map
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        v1 = 0
        v2 = 1
        v3 = 2
        ax.scatter(DM[:,v1], DM[:,v2], DM[:,v3], c=enum, cmap='gist_rainbow', linewidths=lw, s=s, edgecolor='k')    
        ax.set_xlabel('psi %s' % v1)
        ax.set_ylabel('psi %s' % v2)
        ax.set_zlabel('psi %s' % v3)
        ax.view_init(elev=-90, azim=90)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
        plt.show()
    
    
else: #batch print all PD embeddings to files in current directory

    # determine total number of PDs:
    PDs = 0
    for root, dirs, files in os.walk(dmDir):
        for file in sorted(files):
            if not file.startswith('.'): #ignore hidden files
                if file.startswith('gC_trimmed_psi_prD'):
                    PDs += 1
    print('total PDs:', PDs)

    # ready figure directory
    outDir = os.path.join(pyDir, 'Embeddings_%s' % projName)
    if os.path.exists(outDir):
        shutil.rmtree(outDir)
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    for PD in range(0,PDs):
        fname = open(os.path.join(dmDir,'gC_trimmed_psi_prD_%s' % (PD)), 'rb')
        data = pickle.load(fname)
        DM = data['psi']
        
        s = 20
        lw = .5
        idx = 1
        print(PD+1, np.shape(DM)[0])
        m = np.shape(DM)[0]
        enum = np.arange(1,m+1)
        fig = plt.figure() 
        dimRows = 6 #4
        dimCols = 6 #5
        plt.suptitle('PD: %s | Occupancy: %s' % (PD+1, np.shape(DM)[0]))
    
        for v1 in range(1,dimRows+1):
            for v2 in range(v1, v1+dimCols):
                plt.subplot(dimRows, dimCols, idx)
                plt.scatter(DM[:,v1-1], DM[:,v2-1], s=s, edgecolor='k', linewidths=.1, zorder=1)
                plt.xlabel(r'$\Psi_{%s}$' % (v1), fontsize=8, labelpad=2)
                plt.ylabel(r'$\Psi_{%s}$' % (v2), fontsize=8, labelpad=2)
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
                #plt.axis('scaled')
        plt.tight_layout()
        #plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97, wspace=0.25, hspace=0.25)
        fig = plt.gcf()
        fig.savefig(os.path.join(outDir,'Embedding_%s.png' % (PD+1)), dpi=200)
        #plt.show()
        plt.clf()
