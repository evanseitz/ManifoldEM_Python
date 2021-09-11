from __future__ import print_function
import qMult_bsx
import spider
import star
import numpy as np
import logging,util
import p

'''
Copyright (c) UWM, Ali Dashti 2016 (matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
Copyright (c) Columbia University Sonya Hanson 2018 (python version)    
'''

#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)

def get_from_relion(align_star_file, flip):

      df = star.parse_star(align_star_file, keep_index=False)

      try:
            U = df['rlnDefocusU'].values
            V = df['rlnDefocusV'].values
      except:
            print("missing defocus")
            exit(1)

      if 'rlnOriginX' in df.columns and 'rlnOriginY' in df.columns:
            shx = df['rlnOriginX'].values
            shy = df['rlnOriginY'].values
      else:
            shx = U*0.
            shy = shx
      sh = (shx,shy)

      try:
            phi = np.deg2rad(df['rlnAngleRot'].values)
            theta = np.deg2rad(df['rlnAngleTilt'].values)
            psi = np.deg2rad(df['rlnAnglePsi'].values)
      except:
          print("missing Euler angles")
          exit(1)
      try:
            p.EkV = df['rlnVoltage'].values[0]
            p.Cs = df['rlnSphericalAberration'].values[0]
            p.AmpContrast = df['rlnAmplitudeContrast'].values[0]
      except:
            print('missing microscope parameters')
            exit(1)

      qz, qy, qzs = util.eul_to_quat(phi, theta, psi, flip)
      q = qMult_bsx.op(qzs, qMult_bsx.op(qy, qz))

      return (sh,q,U,V)

def get_q(align_param_file, phiCol, thetaCol, psiCol, flip):

      # read the angles
      align = spider.parse(align_param_file)
      phi = np.deg2rad(align[:, phiCol])
      theta = np.deg2rad(align[:, thetaCol])
      psi = np.deg2rad(align[:, psiCol])
      qz, qy, qzs = util.eul_to_quat(phi, theta, psi, flip)
      q = qMult_bsx.op(qzs, qMult_bsx.op(qy, qz))
      return q

def get_df(align_param_file,dfCol):
      # read df
      align = spider.parse(align_param_file)
      if len(dfCol) == 1:
         df = align[:, dfCol]
      if len(dfCol) == 2:
         df = (align[:, dfCol[0]] + align[:, dfCol[1]])/2
       
      return df

def get_shift(align_param_file,shx_col,shy_col):
    # read the x-y shifts
    align = spider.parse(align_param_file)
    sh = (align[:, shx_col]*0, align[:, shy_col]*0)
    return sh

