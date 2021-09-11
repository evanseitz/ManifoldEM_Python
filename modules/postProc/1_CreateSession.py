#run via: chimera 1_CreateSession.py
svd = False #user parameter; True if SVD was performed on volume files in '/post/1_vol'

################################################################################
# GENERATE CHIMERA SESSION #
## Copyright (c) Columbia University Evan Seitz 2019
## Copyright (c) UWM Ali Dashti 2019 
################################################################################

import os
from chimera import runCommand as rc

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
parDir = os.path.abspath(os.path.join(pyDir, os.pardir))
if svd == False:
    volPath = parDir + '/1_vol'
else:
    volPath = parDir + '/2_svd'

states=(1,51) #indices of NLSA states

fnames = []
for i in range(*states):
    if svd:
        fnames.append(os.path.join(volPath, 'SVDimgsRELION_1_%s_of_50.mrc' % i))
    else:
        fnames.append(os.path.join(volPath, 'EulerAngles_1_%s_of_50.mrc' % i))

idx = 0	
for f in fnames:
    rc('open' + f)
    rc('volume #%d step 4' % idx)
    rc("volume #%d hide" % idx)
    idx += 1
