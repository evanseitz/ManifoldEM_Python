import pandas as pd

'''
Copyright (c) Columbia University Sonya Hanson 2018
Copyright (c) Columbia University Hstau Liao 2019
''' 

def write_star(star_file, traj_file, df):
    with open(star_file,'w') as text_file:   
        text_file.write('\ndata_ \n \nloop_ \n \n_rlnImageName #1 \n_rlnAnglePsi #2 \n_rlnAngleTilt #3 \n_rlnAngleRot #4 \n')
        for i in range(len(df)):
            text_file.write('%s@%s %s %s %s\n' %(i+1,traj_file,df.psi[i],df.theta[i],df.phi[i])) 
    
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
