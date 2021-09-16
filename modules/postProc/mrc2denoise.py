import os,sys
import numpy as np
import scipy.misc
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
from pylab import plot, loadtxt, imshow, show, xlabel, ylabel


'''
first, source 'ManifoldEM' conda environment;
then run bash script (which will access this one) via: 'sh mrc2denoise.sh'
denoising type: "Gaussian" or "median"
kernel/window size: k=5
beginnig and ending frames affected: f=5 

'''
################################################################################
# SIMPLE DENOISING GAUSSIAN OR MEDIAN #
## Copyright (c) Columbia University Hstau Liao 2019
################################################################################

def op(proj_name, user_dir, type, f, k):
    p.init()
    p.proj_name = proj_name
    p.user_dir = user_dir
    set_params.op(1)
    outputsDir = os.path.join(p.user_dir, 'outputs_%s' % proj_name)
    f = int(f)
    k = int(k)
    # range of frames to be filtered
    range_init = range(f)
    range_end = range(p.nClass - f, p.nClass)

    for bin in range_init + range_end:
        rec_file = os.path.join(outputsDir, 'post/1_vol/EulerAngles_{}_{}_of_{}.mrc'.format(p.trajName, bin + 1, p.nClass))
        with mrcfile.open(rec_file) as mrc:
            vol = mrc.data
            vol = vol.astype(np.float64)
        if type == 'Gaussian':
            vol = ndimage.gaussian_filter(vol,k)
        elif type == 'median':
            vol = ndimage.median_filter(vol, k)
        else:
            continue
        vol = vol.astype(np.float32)
        rec1_file = os.path.join(outputsDir, 'post/4_denoise/DenoiseimgsRELION_{}_{}_of_{}.mrc'.format(p.trajName, bin + 1, p.nClass))
        mrc = mrcfile.new(rec1_file)
        mrc.set_data(vol)

if __name__ == '__main__':
    mainDir = sys.argv[2]
    modDir = os.path.join(mainDir, 'modules')
    sys.path.append(modDir)
    import p
    import mrcfile
    import svdRF
    import set_params
    op(sys.argv[1],sys.argv[2],
       sys.argv[3],sys.argv[4],sys.argv[5]) #enter the params file name
