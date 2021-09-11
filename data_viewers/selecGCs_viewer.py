import sys, os
import pickle

#################################################################################
# VIEWER FOR TESSELLATION OUTPUTS
#################################################################################
# HOW TO USE:
## data_viewers folder must be in same parent directory as outputs_{project_name}
## give name of project (below) via 'projName' variable
## load 'ManifoldEM' environment
## change total number of PDs via `PDs` variable
#################################################################################
# Copyright (c) Columbia University Evan Seitz 2020
#################################################################################
# PARAMETERS:
## ['CG1', 'CG', 'nG', 'q', 'df', 'S2', 'S20', 'sh', 'NC']
#################################################################################

projName = 'untitled'
parDir = os.path.abspath('..')
outDir = os.path.join(parDir, 'outputs_%s' % projName)
PDs = 400 #total number of PDs; if unknown, make arbitrarily large

if 1: #number of particles per PD
    for i in range(0,PDs):
        fname = open(os.path.join(outDir,'selecGCs'), 'rb')
        data = pickle.load(fname)

        try:
            fname = open(os.path.join(outDir,'selecGCs'), 'rb')
            data = pickle.load(fname)
            print('%s: %s' % (i,len((data['CG'][i]))))
        except Exception as e:
            pass
    
if 0: #x,y,z coordinates
    for i in range(0,PDs):
        try:
            fname = open(os.path.join(outDir,'selecGCs'), 'rb')
            data = pickle.load(fname)
            print((data['S20'][i]))
        except Exception as e:
            pass


