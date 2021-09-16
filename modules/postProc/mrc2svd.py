import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import plot, loadtxt, imshow, show, xlabel, ylabel
import scipy.misc

#first, source 'ManifoldEM' conda environment;
#then run bash script (which will access this one) via: 'sh mrc2svd.sh'
Topo_list = [0, 1] #list of eigenvectors to retain; [0,1] for first 2... [0,1,2] for first 3, etc.

################################################################################
# SINGULAR VALUE DECOMPOSITION #
## Copyright (c) UWM Ghoncheh Mashayekhi 2019
## Copyright (c) Columbia University Evan Seitz 2019
## Copyright (c) Columbia University Hstau Liao 2019
################################################################################

def op(svd_dir, proj_name, user_dir):
    p.init()
    p.proj_name = proj_name
    p.user_dir = user_dir
    set_params.op(1)
    outputsDir = os.path.join(p.user_dir, 'outputs_%s' % proj_name)
    
    #b = np.zeros((p.nPix**3,p.nClass),dtype=np.float32)
    b_Out = os.path.join(outputsDir, 'post/2_svd/bins.dat')
    b = np.memmap(b_Out, dtype='float32', mode='w+', shape=(p.nPix**3,p.nClass))
    
    print('Loading %s volumes...' % p.nClass)
    for bin in range(p.nClass):
        rec_file = os.path.join(outputsDir, 'post/1_vol/EulerAngles_{}_{}_of_{}.mrc'.format(p.trajName, bin + 1, p.nClass))
        with mrcfile.open(rec_file) as mrc:
            vol = mrc.data
            b[:,bin] = vol.flatten()
            mrc.close()
            print('Input volume:', (bin+1))
    topoNum = 8 #number of topos considered
    print('Performing SVD...')
    U, S, V = svdRF.op(b)
    print('SVD complete. Preparing volumes...')
    sdiag = np.diag(S)

    plt.plot(sdiag**2) #S is square roots of non-zero eigenvalues, thus square diagonal of S
    plt.scatter(range(0,50),sdiag**2)
    plt.title('Eigenvalue Spectrum')
    plt.xlabel(r'$\mathrm{\Psi}$')
    plt.ylabel(r'$\mathrm{\lambda}$', rotation=0)
    #plt.show()
    plt.savefig(svd_dir + '.png', bbox_inches='tight')
    
    i1 = 0
    Npixel = p.nPix**3
    ConOrder = 1
    Topo_mean = np.ones((topoNum,Npixel)) * np.Inf
    for ii in range(topoNum): 
        # s = s + 1  needed?
        Topo = np.ones((Npixel, ConOrder)) * np.Inf
        for k in range(ConOrder):
            Topo[:, k] = U[k * Npixel: (k + 1) * Npixel, ii]
        Topo_mean[ii,:] = np.mean(Topo, axis=1)
    Topo_mean = Topo_mean.reshape((topoNum,p.nPix,p.nPix,p.nPix))
    Topo_mean = Topo_mean.astype(np.float32)
    
    #ConImgT = np.zeros((max(U.shape), p.nClass), dtype='float64')
    ConImgT_Out = os.path.join(outputsDir, 'post/2_svd/ConImgT.dat')
    ConImgT = np.memmap(ConImgT_Out, dtype='float64', mode='w+', shape=(max(U.shape), p.nClass))
    
    for i in Topo_list:
        # %ConImgT = U(:,i) *(sdiag(i)* V(:,i)')*psiC';
        ConImgT = ConImgT + np.matmul(U[:, i].reshape(-1, 1), sdiag[i] * (V[:, i].reshape(1, -1)))
    ConImgT=ConImgT.T.astype(np.float32)
    ConImgT=ConImgT.reshape((p.nClass,p.nPix,p.nPix,p.nPix))

    #ConImgT = ConImgT.T
    #ConImgT = ConImgT.reshape((p.nClass,p.nPix,p.nPix,p.nPix))

    print('Outputting %s volumes...' % p.nClass)
    for bin in range(p.nClass):
        rec1_file = os.path.join(outputsDir, 'post/2_svd/SVDimgsRELION_{}_{}_of_{}.mrc'.format(p.trajName, bin + 1, p.nClass))
        mrc = mrcfile.new(rec1_file)
        mrc.set_data(ConImgT[bin,:,:,:])
        print('Output volume:', (bin+1))

    for t in range(topoNum):
        toporec_file = os.path.join(outputsDir, 'post/2_svd/SVDTOPOimgsRELION_{}_{}_of_{}.mrc'.format(p.trajName, t + 1, p.nClass))
        mrc = mrcfile.new(toporec_file)
        mrc.set_data(Topo_mean[t,:,:,:])
    
    # remove temporary (large) files saved via memmap:
    os.remove(os.path.join(outputsDir, 'post/2_svd/bins.dat'))
    os.remove(os.path.join(outputsDir, 'post/2_svd/ConImgT.dat'))
        

if __name__ == '__main__':
    mainDir = sys.argv[2]
    modDir = os.path.join(mainDir, 'modules')
    sys.path.append(modDir)
    import p
    import mrcfile
    import svdRF
    import set_params
    op(os.path.splitext(sys.argv[0])[0], sys.argv[1], sys.argv[2]) #enter the params file name
