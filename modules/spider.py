import sys
import numpy as np
import p
'''
Copyright (c) Columbia University Hstau Liao 2018 (python version)    
'''

def parse(filename):
    ''' parsing a SPIDER DOC file
     :Parameters:
     
     filename : str
                Name of the file
    '''    
    table=[]
    with open(filename, 'r') as fin:

       p.num_part = 0
       for line in fin:
          line1 = line.strip()
          words = line1.split()
          #if words[0] != ';':
          if words[0].find(';') == -1:
              p.num_part += 1
              words = [float(x) for x in words]
              table.append(words)
    table = np.array(table)
    # skip the second column
    table = np.hstack((table[:,0].reshape(-1,1),table[:,2:]))     
    return table         

def write(filename,table):
    with open(filename, 'w') as fout:
       #key = np.arange(len(table).reshape(len(table),1)
       count = 1
       for ind in table:
           print >>fout, '{:5d}'.format(count),'{:2d}'.format(1),'{:5d}'.format(int(ind)),
           count += 1
           #print >>fout, count,1,'{:16.8f}'.format(ind),
           
           print >>fout
          #   print >>fout, '{:16.8f}'.format(val),
             
                 
if __name__ == '__main__':
    filename = sys.argv[1]   
    table = parse(filename)
    #print numpy.array(table)
      
