import logging, sys
import myio
import numpy as np
from subprocess import call
import gc
#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

'''
Copyright (c) Columbia University Evan Seitz 2019
'''


def op(clust, PrD):
    import p
    dist_file = '{}prD_{}'.format(p.dist_file, PrD)
    data = myio.fin1(dist_file)

    imgAll = data['imgAll']
    imgSize = np.shape(imgAll)[0]
    boxSize = np.shape(imgAll)[1]

    imgAvg = np.zeros(shape=(boxSize,boxSize), dtype=float)

    for i in clust:
        imgAvg += imgAll[i]

    return imgAvg
