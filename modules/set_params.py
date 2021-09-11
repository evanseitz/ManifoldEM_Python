'''
Copyright (c) Columbia University Evan Seitz 2019    
Copyright (c) Columbia University Hstau Liao 2019    
'''

def op(do): # p.tess_file is known
    import p
    import myio
    import pickle
    import os

    if do == 1: #via 'set_params.op(1)': to retrieve data stored in parameters file
        data = myio.fin1('params_{}.pkl'.format(p.proj_name))
        params = data['params']

        if 'proj_name' in params.keys():
            p.proj_name = params['proj_name']
        if 'user_dir' in params.keys():
            p.user_dir = params['user_dir']
        if 'resProj' in params.keys():
            p.resProj = params['resProj']
        if 'relion_data' in params.keys():
            p.relion_data = params['relion_data']
        if 'ncpu' in params.keys():
            p.ncpu = params['ncpu']
        if 'machinefile' in params.keys():
            p.machinefile = params['machinefile']
        if 'avg_vol_file' in params.keys():
            p.avg_vol_file = params['avg_vol_file']
        if 'img_stack_file' in params.keys():
            p.img_stack_file = params['img_stack_file']
        if 'align_param_file' in params.keys():
            p.align_param_file = params['align_param_file']
        if 'mask_vol_file' in params.keys():
            p.mask_vol_file = params['mask_vol_file']
        if 'num_part' in params.keys():
            p.num_part = params['num_part']
        if 'Cs' in params.keys():
            p.Cs = params['Cs']
        if 'EkV' in params.keys():
            p.EkV = params['EkV']
        if 'AmpContrast' in params.keys():
            p.AmpContrast = params['AmpContrast']
        if 'nPix' in params.keys():
            p.nPix = params['nPix']
        if 'pix_size' in params.keys():
            p.pix_size = params['pix_size']
        if 'obj_diam' in params.keys():
            p.obj_diam = params['obj_diam']
        if 'resol_est' in params.keys():
            p.resol_est = params['resol_est']
        if 'ap_index' in params.keys():
            p.ap_index = params['ap_index']
        if 'ang_width' in params.keys():
            p.ang_width = params['ang_width']
        if 'sh' in params.keys():
            p.sh = params['sh']
        if 'PDsizeThL' in params.keys():
            p.PDsizeThL = params['PDsizeThL']
        if 'PDsizeThH' in params.keys():
            p.PDsizeThH = params['PDsizeThH']
        if 'S2rescale' in params.keys():
            p.S2rescale = params['S2rescale']
        if 'S2iso' in params.keys():
            p.S2iso = params['S2iso']
        if 'numberofJobs' in params.keys():
            p.numberofJobs = params['numberofJobs']
        if 'num_psis' in params.keys():
            p.num_psis = params['num_psis']
        if 'dim' in params.keys():
            p.dim = params['dim']
        if 'temperature' in params.keys():
            p.temperature = params['temperature']

    elif do == 0: #via 'set_params.op(0)': to update values within parameters file
        params = dict()
        if hasattr(p, 'proj_name'):
            extra = dict(proj_name=p.proj_name)
            params.update(extra)
        if hasattr(p, 'user_dir'):
            extra = dict(user_dir=p.user_dir)
            params.update(extra)
        if hasattr(p, 'resProj'):
            extra = dict(resProj=p.resProj)
            params.update(extra)
        if hasattr(p, 'relion_data'):
            extra = dict(relion_data=p.relion_data)
            params.update(extra)
        if hasattr(p, 'ncpu'):
            extra = dict(ncpu=p.ncpu)
            params.update(extra)
        if hasattr(p, 'machinefile'):
            extra = dict(machinefile=p.machinefile)
            params.update(extra)
        if hasattr(p, 'avg_vol_file'):
            extra = dict(avg_vol_file=p.avg_vol_file)
            params.update(extra)
        if hasattr(p, 'img_stack_file'):
            extra = dict(img_stack_file=p.img_stack_file)
            params.update(extra)
        if hasattr(p, 'align_param_file'):
            extra = dict(align_param_file=p.align_param_file)
            params.update(extra)
        if hasattr(p, 'mask_vol_file'):
            extra = dict(mask_vol_file=p.mask_vol_file)
            params.update(extra)
        if hasattr(p, 'num_part'):
            extra = dict(num_part=p.num_part)
            params.update(extra)
        if hasattr(p, 'Cs'):
            extra = dict(Cs=p.Cs)
            params.update(extra)
        if hasattr(p, 'EkV'):
            extra = dict(EkV=p.EkV)
            params.update(extra)
        if hasattr(p, 'AmpContrast'):
            extra = dict(AmpContrast=p.AmpContrast)
            params.update(extra)
        if hasattr(p, 'nPix'):
            extra = dict(nPix=p.nPix)
            params.update(extra)
        if hasattr(p, 'pix_size'):
            extra = dict(pix_size=p.pix_size)
            params.update(extra)
        if hasattr(p, 'obj_diam'):
            extra = dict(obj_diam=p.obj_diam)
            params.update(extra)
        if hasattr(p, 'resol_est'):
            extra = dict(resol_est=p.resol_est)
            params.update(extra)
        if hasattr(p, 'ap_index'):
            extra = dict(ap_index=p.ap_index)
            params.update(extra)
        if hasattr(p, 'ang_width'):
            extra = dict(ang_width=p.ang_width)
            params.update(extra)
        if hasattr(p, 'sh'):
            extra = dict(sh=p.sh)
            params.update(extra)
        if hasattr(p, 'PDsizeThL'):
            extra = dict(PDsizeThL=p.PDsizeThL)
            params.update(extra)
        if hasattr(p, 'PDsizeThH'):
            extra = dict(PDsizeThH=p.PDsizeThH)
            params.update(extra)
        if hasattr(p, 'S2rescale'):
            extra = dict(S2rescale=p.S2rescale)
            params.update(extra)
        if hasattr(p, 'S2iso'):
            extra = dict(S2iso=p.S2iso)
            params.update(extra)
        if hasattr(p, 'numberofJobs'):
            extra = dict(numberofJobs=p.numberofJobs)
            params.update(extra)
        if hasattr(p, 'num_psis'):
            extra = dict(num_psis=p.num_psis)
            params.update(extra)
        if hasattr(p, 'dim'):
            extra = dict(dim=p.dim)
            params.update(extra)
        if hasattr(p, 'temperature'):
            extra = dict(temperature=p.temperature)
            params.update(extra)

        myio.fout1('params_{}.pkl'.format(p.proj_name), ['params'], [params])

    elif do == -1: #via 'set_params.op(-1)': to print all values in parameters file
        fname = os.path.join(p.user_dir, 'params_{}.pkl'.format(p.proj_name))
        pickle_off = open(fname, 'rb')
        emp = pickle.load(pickle_off)
        print(emp)

    return
