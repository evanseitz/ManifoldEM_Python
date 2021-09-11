#first, source 'ManifoldEM' conda environment;
#then run via: 'python 3_GenGif.py'
session = 'view1' #name of session saved in Chimera (e.g., 'view1' for view1.py)

################################################################################
# GENERATE CHIMERA GIF #
## Copyright (c) Columbia University Evan Seitz 2019
################################################################################
import os
import imageio
from scipy import misc
from PIL import Image, ImageEnhance
import numpy as np

states=(1,51)

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
imgDir = os.path.join(pyDir, 'views/%s' % (session)) #folder where GIF will be written

imgs = []
for i in xrange(*states):
    if 1: #WITHOUT ENHANCEMENT:
        img = imageio.imread(os.path.join(imgDir, '%s_%s.png' % (session, i)))
        imgs.append(img)

    if 0: #WITH ENHANCEMENT:
        img = Image.open(os.path.join(imgDir, '%s_%s.png' % (session,i)))
        enhancer = ImageEnhance.Contrast(img)
        enhanced_im = enhancer.enhance(1.30)
        imgs.append(np.array(enhanced_im))

imageio.mimsave(imgDir + '/%s.gif' % session, imgs)
