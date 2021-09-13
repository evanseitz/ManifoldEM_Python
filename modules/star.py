import pandas as pd
import set_params


'''
Copyright (c) Columbia University Sonya Hanson 2018
Copyright (c) Columbia University Hstau Liao 2019
''' 

def write_star(star_file, traj_file, df):
    import p
    set_params.op(1)
    
    with open(star_file,'w') as text_file:   
        text_file.write('\ndata_ \n \nloop_ \n \n_rlnImageName #1 \n_rlnAnglePsi #2 \n_rlnAngleTilt #3 \n_rlnAngleRot #4 \n_rlnDetectorPixelSize #5 \n_rlnMagnification #6 \n')
        for i in range(len(df)):
            text_file.write('%s@%s %s %s %s %s %s\n' %(i+1,traj_file,df.psi[i],df.theta[i],df.phi[i],p.pix_size,10000.0)) 
            # Note: DetectorPixelSize and Magnification required by relion_reconstruct; 10000 used here such that we can always put in the user-defined pixel size...
            # ...since it may be obtained via calibration (see user manual); since Pixel Size = Detector Pixel Size [um] / Magnification --> [Angstroms]
'''
This parse_star function is from version 0.1 of pyem by Daniel Asarnow at UCSF
'''

def parse_star(starfile, keep_index=False):
    headers = []
    foundheader = False
    ln = 0
    with open(starfile, 'rU') as f:
        for l in f:
            if l.startswith("_rln"):
                foundheader = True
                lastheader = True
                if keep_index:
                    head = l.rstrip()
                else:
                    head = l.split('#')[0].rstrip().lstrip('_')
                headers.append(head)
            else:
                lastheader = False
            if foundheader and not lastheader:
                break
            ln += 1
    star = pd.read_table(starfile, skiprows=ln, delimiter='\s+', header=None)
    star.columns = headers
    return star
