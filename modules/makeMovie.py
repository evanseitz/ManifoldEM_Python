import logging, sys, time
import numpy as np
import p
import imageio

'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)

'''

def op(IMG1,prD,psinum,fps):
    import matplotlib.pyplot as plt #must be imported within the function (for parallel processing)
    dim = np.floor(np.sqrt(max(IMG1.shape)))  # window size
    dim = int(dim)
    
    gifImgs = []
    
    for i in range(IMG1.shape[1]):
        time.sleep(.01)
        IMG = -IMG1[:, i].reshape(dim, dim)  # an image
        #IMG = IMG.T
        frame = p.out_dir + '/topos/PrD_{}/psi_{}/frame{:02d}'.format(prD + 1, psinum + 1, i)

        fig = plt.figure(frameon=False)
        
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(IMG, cmap=plt.get_cmap('gray'))
        #ax1.set_title('Frame %02d' % (i+1))
        
        fig.savefig(frame, bbox_inches='tight', pad_inches=.1)
        
        ax.clear()
        plt.close()
        gifImgs.append(imageio.imread(frame + '.png'))

    gifDir = p.out_dir + '/topos/PrD_{}/'.format(prD + 1)
    imageio.mimsave(gifDir + 'psi_{}.gif'.format(psinum + 1), gifImgs, subrectangles=False)#,quantizer='nq')

    return None

        
    # call(["mkdir", "-p", 'tmp'])            # create a temp dir for video creation
    #os.chdir('tmp')
    #call(['/home/hstau/anaconda2/bin/ffmpeg', '-framerate', '50', '-i', 'file%02d.png', '-r','30', '-pix_fmt', 'yuv420p','tmp.mp4'])
    #for file_name in glob.glob("*.png"):
    #    os.remove(file_name)
    #os.chdir('../')
    #call(['mv', 'tmp/tmp.mp4', outFile])

