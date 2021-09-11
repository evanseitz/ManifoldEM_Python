import numpy as np

'''
Copyright (c) Columbia University Hstau Liao 2019    
'''

def op(A,b):
    try:
        coeff = np.linalg.lstsq(A, b)[0]
        #coeff = np.linalg.solve(A,b)
    except:
        coeff = np.linalg.solve(A, b)
        #coeff = np.linalg.lstsq(A,b)[0]

    return coeff
