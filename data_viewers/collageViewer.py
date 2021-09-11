import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np

#################################################################################
# GENERATE NLSA MOVIES AS A COLLAGE (.GIF) FOR EACH PD
#################################################################################
# HOW TO USE:
## data_viewers folder must be in same parent directory as outputs_{project_name}
## give name of project (below) via 'projName' variable
## load 'ManifoldEM' environment
## WARNING: this can take up a considerable amount of space (~1e2-1e3 GB)
#################################################################################
# Copyright (c) Columbia University Evan Seitz 2020
#################################################################################

projName = 'untitled'
parDir = os.path.abspath('..')
toposDir = os.path.join(parDir, 'outputs_%s/topos' % (projName))
if not os.path.exists('psiGIFs'):
    os.makedirs('psiGIFs')

def create_collage(width, height, imageList, cols, rows):
    cols = 4
    rows = 2
    thumbnail_width = width//cols
    thumbnail_height = height//rows
    size = thumbnail_width, thumbnail_height
    new_im = Image.new('L', (width, height))
    ims = []
    for p in imageList:
        ims.append(p)
    i = 0
    x = 0
    y = 0
    for col in range(rows):
        for row in range(cols):
            new_im.paste(ims[i], (y, x))
            i += 1
            y += thumbnail_height
        x += thumbnail_width
        y = 0
    return new_im

PDs = 0
for root, dirs, files in os.walk(toposDir):
    for file in sorted(files):
        if not file.startswith('.'): #ignore hidden files
            if file.startswith('class_avg'):
                PDs += 1
print('total PDs:', PDs)

cols = 4
rows = 2
for pd in range(0, PDs):
    print('Generating PrD_%s GIF...' % (pd+1))
    pdDir = os.path.join(toposDir, 'PrD_%s' % (pd+1))
    frames = []
    for fr in range(50):#[1,49]
        psis = []
        for i in range(8):
            try:
                psi = Image.open(os.path.join(pdDir,'psi_%s/frame%02d.png' % (i+1, fr)))
            except:
                psi = Image.new('L', (dim,dim))
            dim = np.shape(psi)[0]
            draw = ImageDraw.Draw(psi)
            font = ImageFont.truetype('arial.ttf', 20)
            draw.text((dim*.05,dim*.05), 'PD_%s_Psi_%s' % (pd+1, i+1), fill='white', font=font)
            draw.text((dim*.1,dim*.1), 'frame_%02d' % fr, fill='white', font=font)
            psis.append(psi)
        
        frame = create_collage(dim*cols, dim*rows, psis, cols, rows)
        frames.append(frame)
        
    frames[0].save('psiGIFs/PrD_%s.gif' % (pd+1), format='GIF', append_images=frames[1:], save_all=True, duration=.00001, loop=0)
