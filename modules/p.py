import os
import numpy as np
from subprocess import call
from shutil import copyfile

'''
Copyright (c) Columbia University Evan Seitz 2019
Copyright (c) Columbia University Hstau Liao 2019
Copyright (c) Columbia University Suvrajit Maji 2019   
'''


def init():
    # resume project:
    global proj_name #user-defined project name
    global resProj #GUI gatekeeping for resuming previous project
    '''
    resProj structure:
        0: default; new project
        1: user has confirmed Data.py entries, but not yet started (or partially started) GetDistancesS2.py
        2: GetDistances.py complete, possible partial-progress on manifoldAnalysis.py
        3: manifoldAnalysis.py complete, possible partial-progress on psiAnalysis.py
        4: psiAnalysis.py complete, possible partial-progress on NLSAmovie.py
        5: NLSAmovie.py complete
        6: PD anchors chosen/saved, "Compile" button clicked, possible partial-progress on FindReactionCoord.py
        7: FindReactionCoord.py complete, possible partial-progress on EL1D.py
        8: EL1D.py complete, possible partial-progress on PrepareOutputS2.py
        9: PrepareOutputS2.py complete
        '''

    # project format:
    global relion_data #False for not-relion (Spider) input-data format
    global phiCol, thetaCol, psiCol, dfCol, shx_col, shy_col #for reading Spider align file

    # computational resources:
    global ncpu #number of CPUs
    global machinefile #via arg: '--mpi path/to/machinefile' (default False for non-MPI run)
    global eps
    eps = 1e-10 #small fraction to be added if divide-by-zero errors occur

    # microscopy parameters:
    global avg_vol_file #average volume file (e.g., .mrc)
    global img_stack_file #image stack file (e.g., .mrcs)
    global align_param_file #alignment file (e.g., .star)
    global mask_vol_file #mask volume file (e.g., .mrc)
    global num_part #total number of particles in stack
    global Cs #Spherical Aberration [mm] (from alignment file)
    global EkV #Voltage [kV] (from alignment file)
    global AmpContrast #Amplitude Contrast [ratio](from alignment file);
    global gaussEnv
    gaussEnv = np.inf #envelope of CTF
    global nPix #window size of image (e.g., for 100x100 image, nPix=100)
    global pix_size #pixel size of image [Angstroms] (known via rln_DetectorPixelSize*10e6 / rln_Magnification)
    global obj_diam #diameter of macromolecule [Angstroms]
    global resol_est #estimated resolution [Angstroms]
    global ap_index #aperture index {1,2,3...}; increases tessellated bin size
    global ang_width #angle width (via: ap_index * resol_est / obj_diam)
    global sh #Shannon angle (pix_size / obj_diam)

    # tessellation binning:
    #PDsizeThL and PDsizeThH defined by default in Manifold_GUI.py
    global PDsizeThL #minimum required snapshots in a tessellation for it be admitted
    global PDsizeThH #maximum number of snapshots that will be considered within each tessellation
    global S2rescale #proper scale ratio between S2 sphere and .mrc volume for visualizations
    global S2iso #proper isosurface level of .mrv volume for vizualiaztion (as chosen by user)
    global numberofJobs #total number of bins to consider for manifold embedding

    # eigenfunction parameters:
    global num_eigs, num_psiTrunc, num_psis, tune, rad, conOrderRange, sizeToConOrderRatio
    num_eigs = 15 #number of highest-eigenvalue eigenfunctions to consider in total (max entry of eigenvalue spectrum)
    num_psiTrunc = 8 #number of eigenfunctions for truncated views
    tune = 3 #diffusion map tuning; this needs to be automated
        #tune automation suggestion (Ali): larger tune = smaller gaussian width; turns data into
        #islands/chunks (can't see long-range patterns); a good tune should form a 'good' psirec parabola.
        #as well, you can keep using tune to get rid of outliers in data; you want the number of outliers
        #to be around 10%; if higher than this, tune needs to be changed.
    rad = 5 #manifold pruning
    conOrderRange = 50 #coarse-graining factor of energy landscape

    # NLSA movie parameters:
    global fps, nC
    fps = 5 #frames per second of NLSA movie (currently inactive)

    # energy landscape parameters:
    global dim #user-defined number of dimensions (reaction coordinates); {1,2}
    global temperature #user-defined pre-quenching temperature of experiments
    global num_ang, isEq, trajName, isTrajClosed, boundCond, nClass, nClass2D, xchosed
    num_ang = 180 #number of tomographic projections in 2D
    isEq = 0 #equalization, used during tau (T/F)
    trajName = '1' #filename for exported (2D) trajectories
    isTrajClosed = 1 #2D ELS parameter
    boundCond = 0 #2D ELS parameter
    nClass = 50 #number of states partitioned within each 1D reaction coordinate; results in a 50x1 1D ELS
    nClass2D = 250 #number of states partitioned within 2D tau(theta) cuts; results in a 176x176 2D ELS
        #nClass2D (above) still needs to replace 'nC' in Matlab's ELshow.m and OMprofiles.m
    global EL1D #array for 1D Energy Landscape
    global selPts_1D, selPts_2D, width_1D, width_2D, leastPts
    width_1D = 1 #user-defined width of trajectory in 1D energy path
    width_2D = 1 #user-defined width of trajectory in 2D energy path
    selPts_2D = []  #all 2D energy landscape points between user 'Resets' (from current pixel width)
    leastPts = []  #all points from least-action computation
    global hUn #occupancy map

    # reaction coordinates parameters:
    global getOpticalFlow, getAllEdgeMeasures, anch_list, trash_list, opt_movie, opt_mask_type, opt_mask_param
    getOpticalFlow = 1 #default True to compute optical flow vectors
    getAllEdgeMeasures = 1 #default True to compute edge measures
    anch_list = [] #user-defined PD anchors for Belief Propagation
    trash_list = [] #user-defined PD removals to ignore via final compile [binary list, 1 entry/PD]
    opt_movie = dict(printFig=0, OFvisual=0, visual_CC=0, flowVecPctThresh=95) #default False: won't save movies to file
    opt_mask_type = int(0) #0:None, 1:Annular, 2:Volumetric
    opt_mask_param = int(0) #for either none, radius (Int), or iso(Int)

    return None

def create_dir():
    # input and output directories and files
    global user_dir,dist_dir,dist_prog,psi_dir,psi_prog,psi2_dir,psi2_prog,\
        movie2d_dir,EL_dir,EL_prog,\
        tau_dir,OM_dir,Var_dir,NLSA_dir,traj_dir,bin_dir,relion_dir,\
        CC_dir,CC_OF_dir, CC_meas_dir, CC_meas_prog, out_dir, \
        post_dir, vol_dir, svd_dir, anim_dir

    dist_dir = os.path.join(user_dir, 'outputs_{}/distances/'.format(proj_name))
    call(["mkdir", "-p", dist_dir])
    dist_prog = os.path.join(dist_dir, 'progress/')
    call(['mkdir', '-p', dist_prog])

    psi_dir = os.path.join(user_dir, 'outputs_{}/diff_maps/'.format(proj_name))
    call(["mkdir", "-p", psi_dir])
    psi_prog = os.path.join(psi_dir, 'progress/')
    call(['mkdir', '-p', psi_prog])

    psi2_dir = os.path.join(user_dir, 'outputs_{}/psi_analysis/'.format(proj_name))
    call(["mkdir", "-p", psi2_dir])
    psi2_prog = os.path.join(psi2_dir, 'progress/')
    call(['mkdir', '-p', psi2_prog])

    EL_dir = os.path.join(user_dir, 'outputs_{}/ELConc{}/'.format(proj_name,conOrderRange))
    call(["mkdir", "-p", EL_dir])
    EL_prog = os.path.join(EL_dir, 'progress/')
    call(['mkdir', '-p', EL_prog])

    OM_dir = os.path.join(user_dir,'{}OM/'.format(EL_dir))
    call(["mkdir", "-p", OM_dir])

    Var_dir = os.path.join(user_dir,'outputs_{}/Var/'.format(proj_name))
    call(["mkdir", "-p", Var_dir])
    traj_dir = os.path.join(user_dir,'outputs_{}/traj/'.format(proj_name))
    call(["mkdir", "-p", traj_dir])

    bin_dir = os.path.join(user_dir,'outputs_{}/bin/'.format(proj_name))
    call(["mkdir", "-p", bin_dir])
    relion_dir = os.path.join(user_dir,'outputs_{}/bin/'.format(proj_name))
    call(["mkdir", "-p", relion_dir])

    CC_dir = os.path.join(user_dir,'outputs_{}/CC/'.format(proj_name))
    call(["mkdir", "-p", CC_dir])
    CC_OF_dir = os.path.join(user_dir,'outputs_{}/CC/CC_OF/'.format(proj_name))
    call(["mkdir", "-p", CC_OF_dir])
    CC_meas_dir = os.path.join(user_dir,'outputs_{}/CC/CC_meas/'.format(proj_name))
    call(["mkdir", "-p", CC_meas_dir])
    CC_meas_prog = os.path.join(CC_meas_dir, 'progress/')
    call(['mkdir', '-p', CC_meas_prog])

    #################
    # post-processing:
    post_dir = os.path.join(user_dir, 'outputs_{}/post/'.format(proj_name))
    vol_dir = os.path.join(post_dir, '1_vol')
    svd_dir = os.path.join(post_dir, '2_svd')
    anim_dir = os.path.join(post_dir, '3_anim')
    call(["mkdir", "-p", post_dir])
    call(["mkdir", "-p", vol_dir])
    call(["mkdir", "-p", svd_dir])
    call(["mkdir", "-p", anim_dir])
    pp_dir = os.path.join(os.path.sep, user_dir, 'modules', 'postProc')

    post1 = os.path.join(pp_dir, 'mrcs2mrc.sh')
    post2 = os.path.join(pp_dir, '1_CreateSession.py')
    post3 = os.path.join(pp_dir, '2_GenMovie.py')
    post4 = os.path.join(pp_dir, '3_GenGif.py')
    post5 = os.path.join(pp_dir, 'mrc2svd.py')
    post6 = os.path.join(pp_dir, 'mrc2svd.sh')
    post7 = os.path.join(pp_dir, 'mrc2denoise.py')

    copyfile(post1, os.path.join(vol_dir, 'mrcs2mrc.sh'))
    copyfile(post2, os.path.join(anim_dir, '1_CreateSession.py'))
    copyfile(post3, os.path.join(anim_dir, '2_GenMovie.py'))
    copyfile(post4, os.path.join(anim_dir, '3_GenGif.py'))
    copyfile(post5, os.path.join(svd_dir, 'mrc2svd.py'))
    copyfile(post6, os.path.join(svd_dir, 'mrc2svd.sh'))
    copyfile(post7, os.path.join(svd_dir, 'mrc2denoise.py'))

    #################

    out_dir = os.path.join(user_dir,'outputs_{}/'.format(proj_name))
    call(["mkdir", "-p", out_dir])
    call(["mkdir", "-p", out_dir+'/topos/'])
    call(["mkdir", "-p", out_dir + '/topos/Euler_PrD/'])

    global dist_file,psi_file,psi2_file,\
        movie2d_file,EL_file,tau_file,OM_file,OM1_file,Var_file,rho_file,\
        remote_file,NLSA_file,traj_file,CC_file,CC_OF_file,CC_meas_file,\
        CC_graph_file,ref_ang_file,ref_ang_file1,tess_file,nowTime_file

    tess_file = os.path.join(user_dir, 'outputs_{}/selecGCs'.format(proj_name))
    nowTime_file = os.path.join(user_dir, 'outputs_{}/nowTime'.format(proj_name))
    dist_file = '{}/IMGs_'.format(dist_dir)
    psi_file = '{}/gC_trimmed_psi_'.format(psi_dir)
    psi2_file = '{}/S2_'.format(psi2_dir)
    EL_file = '{}/S2_'.format(EL_dir)
    OM_file = '{}/S2_'.format(OM_dir)
    OM1_file = '{}/S2_'.format(OM_dir)
    Var_file = '{}/S2_'.format(Var_dir)
    rho_file = '{}/rho'.format(OM_dir)
    remote_file = '{}/rem_'.format(Var_dir)
    traj_file = '{}/traj_'.format(traj_dir)
    CC_graph_file = '{}graphCC'.format(CC_dir)
    CC_OF_file = '{}OF_prD_'.format(CC_OF_dir)
    CC_meas_file = '{}meas_edge_prDs_'.format(CC_meas_dir)
    CC_file = '{}CC_file'.format(CC_dir)
    ref_ang_file = '{}/topos/Euler_PrD/PrD_map.txt'.format(out_dir)
    ref_ang_file1 = '{}/topos/Euler_PrD/PrD_map1.txt'.format(out_dir)

    return None
