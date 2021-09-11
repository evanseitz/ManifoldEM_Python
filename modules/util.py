import numpy as np
tiny = 1e-10
import logging, sys
import myio
import pickle

'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
'''

#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

def hist_match(source, template):  # by ali_m
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def histeq(src,thist): # by Zagurskin; does not work well,
    nbr_bins = len(thist)
    bins = np.linspace(0, 1, nbr_bins + 1)
    # hist, bins = np.histogram(src.flatten(), nbr_bins, normed=True)
    hist, bb = np.histogram(src.flatten(), bins)  # nbr_bins, normed=True)

    cdfsrc = hist.cumsum()  # cumulative distribution function
    cdfsrc = (nbr_bins * cdfsrc / cdfsrc[-1]).astype(np.uint8)  # normalize

    cdftint = thist.cumsum()  # cumulative distribution function
    cdftint = (nbr_bins * cdftint / cdftint[-1]).astype(np.uint8)  # normalize

    h2 = np.interp(src.flatten(), bins[:-1], cdfsrc)
    h3 = np.interp(h2, cdftint, bins[:-1])

    return h3

def eul_to_quat(phi, theta, psi, flip=True):
    try:
        assert (len(phi) > 0 and len(theta) > 0 and len(psi) > 0)
    except AssertionError:
        _logger.error('subroutine eul_to_quat: some Euler angles are missing')
        _logger.exception('subroutine eul_to_quat: some Euler angles are missing')
        raise
        sys.exit(1)

    zros = np.zeros(phi.shape[0])
    qz = np.vstack((np.cos(phi / 2), zros, zros, -np.sin(phi / 2)))
    qy = np.vstack((np.cos(theta / 2), zros, -np.sin(theta / 2), zros))
    sp = np.sin(psi / 2)
    if flip:
        sp = -sp
    qzs = np.vstack((np.cos(psi / 2), zros, zros, sp))
    return (qz, qy, qzs)

def augment(q):
    try:
        assert (q.shape[0] > 3)
    except AssertionError:
        _logger.error('subroutine augment: q has wrong dimensions')
        _logger.exception('subroutine augment: q has wrong diemnsions')
        raise
        sys.exit(1)

    qc = np.vstack((-q[1, :], q[0, :], -q[3, :], q[2, :]))
    q = np.hstack((q, qc))
    # print q.shape
    return q


def useless_loop(sizeToConOrderRatio,tauInDir,xAll,xSelect,psinums,posPaths):
    ang_res = 3
    for x in xSelect:
        gC = xAll[1, x]
        prD = xAll[0, x]
        psinum2 = psinums[1, x]
        psinum1 = psinums[0, x]

        string = '{}gC{}_prD{}_tautotEL'.format(tauInDir,gC,prD)
        data = myio.fin(string,['tautotAll','listBad'])
        tautotAll = data[0]
        listBad = data[1]
        tau = np.zeros((len(tautotAll[0]),ang_res))
        for i in range(ang_res):
            tau[:, i] = tautotAll[i].flatten()
        posPath = posPaths[x]
        nS = len(posPath)
        #ConOrders[x] = max(5, np.floor(nS / sizeToConOrderRatio))

        #taus[x] = tau
        #listBads[x] = listBad

    return


def make_indeces(inputGCs):

    with open(inputGCs, 'rb') as f:
        param = pickle.load(f)
    f.close()

    GCnum = len(param['CGtot'])
    prDs = len(param['CGtot'][0])

    x1 = np.tile(range(prDs), (1, GCnum))
    x2 = np.array([])
    for i in range(GCnum):
        x2 = np.append(x2, np.tile(i, (1, prDs)))
    xAll = np.vstack((x1, x2)).astype(int)
    xSelect = range(xAll.shape[1])

    return xAll,xSelect

def interv(s):
    #return np.arange(-s/2,s/2)
    if s%2 == 0:
        a = -s/2
        b = s/2-1
    else:
        a = -(s-1)/2
        b = (s-1)/2

    return np.linspace(a,b,s)

def filter_fourier(inp, sigma):
    # filter Gauss
    nPix1 = inp.shape[1]
    nPix2 = inp.shape[0]
    X, Y = np.meshgrid(interv(nPix1), interv(nPix2))
    '''
    # nPix1 and nPix2 odd
    if nPix1%2 == 0 and nPix2%2 == 0:
        ab = np.arange(-(nPix2 - 1) / 2,(nPix2 - 1) / 2)
        X, Y = np.meshgrid(interv(nPix1),interv(nPix2))
    elif nPix1%2 == 1 && nPix2%2 == 1:
        aa = np.aranage(-nPix1 / 2,nPix1 / 2 - 1)
        ab = np.aranage(-nPix2 / 2,nPix2 / 2 - 1)
        X, Y = np.meshgrid(aa, ab)
        # nPix1 and nPix2 even
    elif ~mod(nPix1, 2) && mod(nPix2, 2):
        X, Y = meshgrid(-nPix1 / 2:nPix1 / 2 - 1, -(nPix2 - 1) / 2:(nPix2 - 1) / 2)
        # nPix1 even and nPix2 odd
    elif mod(nPix1, 2) && ~mod(nPix2, 2):
        X, Y = meshgrid(-(nPix1 - 1) / 2:(nPix1 - 1) / 2, -nPix2 / 2:nPix2 / 2 - 1)
        # nPix1 odd and nPix2 even
    '''
    Rgrid = nPix2 / 2.
    Q = (1 / Rgrid) * np.sqrt(X ** 2 + Y ** 2)  # Q in units of Nyquist frequency

    N = 4
    G = np.sqrt(1. / (1 + (Q / sigma) ** (2 * N)))  # ButterWorth

    # G = exp(-(log(2) / 2) * (Q / sigmaH). ^ 2);Gaussian

    # Filter images in Fourier space
    G = np.fft.ifftshift(G)
    inp = np.real(np.fft.ifft2(G * np.fft.fft2(inp)))

    return inp

