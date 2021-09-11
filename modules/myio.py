import logging, sys
import pickle
from pickle import UnpicklingError

'''
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
'''

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

#def fin(Filename,arg_list):
    #with open(Filename, 'rb') as f:
        #data = pickle.load(f)
    #f.close()
    #var = []
    #for str in arg_list:
        #var.append(data[str])
    #return var

def fin1(Filename):
    with open(Filename, 'rb') as f:
        try:
            data = pickle.load(f)
            f.close()
            return data
        except UnpicklingError as e:
            return None
        except:
            return None

#def fout(Filename, arg_list):
    #with open(Filename, 'wb') as f:
        #for str in arg_list:
            #pickle.dump(str,f)
    #f.close()

def fout1(Filename, key_list, v_list):
    #print arg_list,locals()
    to_save = dict([(key_list[i], v_list[i]) for i in range(len(key_list))])
    '''
    for arg in arg_list:
        print arg,locals()[arg]
        to_save[arg] = locals()[arg]
        #str = '{}'.format(arg)
        #dt = {str:arg}
        #to_save.update(dt)
    '''
    with open(Filename, 'wb') as f:
        pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def fout2(Filename, dict):
    key_list = []
    v_list =[]
    for key,value in dict.items():
        key_list.append(key)
        v_list.append(value)
    fout1(Filename,key_list,v_list)