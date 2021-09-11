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
# CHECK PROGRESS OF PSI ANALYSIS MODULE (FOR STALLED PDs)
#################################################################################
# HOW TO USE:
## data_viewers folder must be in same parent directory as outputs_{project_name}
## give name of project (below) via 'projName' variable
## load 'ManifoldEM' environment
## output: list of completed Psis and those that may have stalled. If you find
##          that many Psis have stalled after significant processing time,
##          these may represent patternless particle ensembles. We tend to
##          take note of these, then go in and create duplicates of other
##          psi files in their place (renaming as needed); and updating the
##          progress folder. On the next tab of the GUI, you will be able to
##          remove these bad PDs (or ignore some bad Psis within a PD)
#################################################################################
# Copyright (c) Columbia University Evan Seitz 2020
#################################################################################

projName = 'untitled'
parDir = os.path.abspath('..')
psiDir = os.path.join(parDir, 'outputs_%s/psi_analysis' % (projName))
progDir = os.path.join(parDir, 'outputs_%s/psi_analysis/progress' % (projName))

psiPaths = []
for root, dirs, files in os.walk(psiDir):
    for file in sorted(files):
        if not file.startswith('.'): #ignore hidden files
            if file.startswith('S2'):
                psiPaths.append(os.path.join(root, file))

idx = 0
psi1 = []
psi2 = []
psi3 = []
psi4 = []
psi5 = []
psi6 = []
psi7 = []    
psi8 = []
for i in psiPaths:
    if i.endswith('0'):
        psi1.append(i)
    elif i.endswith('1'):
        psi2.append(i)
    elif i.endswith('2'):
        psi3.append(i)
    elif i.endswith('3'):
        psi4.append(i)
    elif i.endswith('4'):
        psi5.append(i)
    elif i.endswith('5'):
        psi6.append(i)
    elif i.endswith('6'):
        psi7.append(i)
    elif i.endswith('7'):
        psi8.append(i)   
    idx+=1
    
print(len(psi1),len(psi2),len(psi3),len(psi4),len(psi5),len(psi6),len(psi7),len(psi8))  
print(idx, idx/8)


fin_PDs = np.zeros(shape=(1678,8), dtype=int) #zeros signify PD_psi entry not complete
for root, dirs, files in os.walk(progDir):
    for file in sorted(files):
        if not file.startswith('.'): #ignore hidden files
            fin_PD, fin_psi = file.split('_')
            fin_PDs[int(fin_PD),int(fin_psi)] = int(1)

missing = []         
for i in range(1678):
    for j in range(0,8):
        if fin_PDs[i,j] == 0:
            ij = 'PD_%s_Psi_%s' % (i,j) 
            missing.append(ij)

for m in missing:           
    print(m)
    
