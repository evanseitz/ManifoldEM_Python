from __future__ import print_function

import sys, os, os.path, time, errno, psutil
from PyQt5.QtWidgets import *
import csv
from subprocess import call
import multiprocessing
import optparse

time_init = time.strftime("%Y%m%d_%H%M%S")
pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
inputDir = os.path.join(pyDir, 'data_input')
modDir = os.path.join(pyDir, 'modules')
CCDir = os.path.join(modDir, 'CC')
BPDir = os.path.join(CCDir, 'BP')
sys.path.append(modDir) #link imports to 'modules' folder
sys.path.append(CCDir) #link imports to 'modules -> CC' folder
sys.path.append(BPDir) #link imports to 'modules -> CC -> BP' folder

from pyface.qt import QtGui, QtCore
os.environ['ETS_TOOLKIT'] = 'qt4'
import sip
sip.setapi('QString', 2)
import threading
import gc #garbage collection
import shutil
import re

import matplotlib
matplotlib.use('Agg') #Qt4Agg

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, Button
import mpl_toolkits.axes_grid1
import matplotlib.path as pltPath
import matplotlib.image as mpimg
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import numpy as np
from pylab import imshow, show, loadtxt, axes
from scipy import stats
import itertools
import Data
import mrcfile
from PIL import Image
import imageio
import cv2
import pickle

import p
p.init() #only initiate once
p.user_dir = pyDir #default
import myio
import GetDistancesS2
import manifoldAnalysis
import psiAnalysis
import NLSAmovie
import embedd
import clusterAvg
import projectMask
import FindConformationalCoord
import EL1D
import backup
import PrepareOutputS2
import set_params

import logging
from traits.api import HasTraits,Any,Instance,on_trait_change,List,Str,Int,Float,Range,Button,Callable,Enum
from traitsui.api import View,Item,HSplit,Group,HGroup,VGroup,ListEditor,TextEditor,RangeEditor,Handler
from mayavi import mlab
from mayavi.core.api import PipelineBase, Engine
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from tvtk.api import tvtk
from tvtk.pyface.api import Scene
from tvtk.common import configure_input_data
#from enthought.pyface.api import GUI #GUI.set_busy()

import warnings
warnings.filterwarnings('ignore', '.*GUI is implemented.*')
warnings.filterwarnings('ignore', '.*Adding an axes using the same*')
warnings.filterwarnings('ignore', '.*Unable to find pixel distance*.') #TauCanvas Linux-only error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', '.*FixedFormatter.*')

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) CU, Evan Seitz 2018-2020
Columbia University
Contact: evan.e.seitz@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

# =============================================================================
# Global assets:
# =============================================================================
progname = 'ManifoldEM'
progversion = '0.2.0-beta'
Beta = True #set to 'True' to disable 2D options; if disabled ('False') within this Beta, demo Ribosome energy landscape can be examined (only) using GUI
#0.1.0: alpha (01.28.19)
#0.2.0: beta (10.01.21) #ZULU update for release
font_header = QtGui.QFont('Arial', 13)
font_standard = QtGui.QFont('Arial', 12)
anchorsMin = 1 #set minimum number of anchors needed for Belief Propagation
p.resProj = 0
p.PDsizeThL = 100
p.PDsizeThH = 2000
if Beta:
    p.dim = 1
else:
    p.dim = 2
p.ncpu = 1
p.num_psis = 8
p.user_dir = pyDir
p.proj_name = time_init
p.temperature = 25
# as a note, the word 'ZULU' is used throughout this document as a bookmark for pending items by E. Seitz.

# =============================================================================
# Command line arguments:
# =============================================================================
parser = optparse.OptionParser()

# --resume user progress via data_inputs.txt location (string):
parser.add_option('--input',
                  help='Name of previous project to resume (i.e., params_<name>.pkl)',
                  type='str', action='store', dest='userInput')
parser.set_defaults(userInput='empty')

# --turn on Message Passing Interface (MPI) via text file location (string):
parser.add_option('--mpi',
                  help='Path (str) to the machinefile for initiating MPI',
                  type='str', action='store', dest='mpiFile')
parser.set_defaults(mpiFile='') #ZULU add to user manual; check is working

# gather command line arguments:
options, args = parser.parse_args()

global userInput
userInput = options.userInput

p.machinefile = options.mpiFile
if options.mpiFile:
    print('')
    print('MPI initiated...')
    print('')

# =============================================================================
# OS identification (if needed):
# =============================================================================
Windows = sys.platform.lower().startswith(('win','microsoft'))
Mac = sys.platform.lower().startswith('darwin')
Linux = sys.platform.lower().startswith(('linux','linux2'))

            
# =============================================================================
# MayaVI viz 1 (P2):
# =============================================================================

class Mayavi_S2(HasTraits): # S2 Orientation Sphere, Electrostatic Potential Map
    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())

    S2_scale_all = List([.2,.4,.6,.8,1.0,1.2,1.4,1.6,1.8,2.0])
    S2_scale = Enum(1.0, values='S2_scale_all')
    display_angle = Button('Display Euler Angles')
    phi = Str
    theta = Str
    display_thresh = Button('PD Thresholding')
    isosurface_level = Range(2,9,3,mode='enum')
    S2_rho = Int
    S2_density_all = List([5,10,25,50,100,250,500,1000,10000,100000]) #needs to match P1.S2_density_all
    S2_density = Enum(S2_rho, values='S2_density_all')

    click_on = 0

    def _phi_default(self):
        return '%s%s' % (0, u"\u00b0")

    def _theta_default(self):
        return '%s%s' % (0, u"\u00b0")

    def _S2_rho_default(self):
        return 100

    def update_S2_params(self):
        self.isosurface_level = int(p.S2iso)
        self.S2_scale = float(P1.S2rescale)
        self.S2_rho = MainWindow.S2_rho
        self.S2_density = int(self.S2_rho)

    def update_S2_density_all(self):
        self.S2_density_all = []
        for i in P1.S2_density_all:
            self.S2_density_all.append(int(i))

    @on_trait_change('display_angle')
    def view_anglesP2(self):
        viewS2 = self.scene1.mlab.view(figure=Mayavi_S2.fig1)
        azimuth = viewS2[0] #phi: 0-360
        elevation = viewS2[1] #theta: 0-180
        zoom = viewS2[2]
        print_anglesP2(azimuth, elevation)

    @on_trait_change('S2_scale, S2_density') #S2 Orientation Sphere
    def update_scene1(self):
        # store current camera info:
        view = self.scene1.mlab.view()
        roll = self.scene1.mlab.roll()

        Mayavi_S2.fig1 = mlab.figure(1, bgcolor=(.5,.5,.5))
        Mayavi_S2.fig2 = mlab.figure(2, bgcolor=(.5,.5,.5))
        
        mlab.clf(figure=Mayavi_S2.fig1)
        
        P1.x1,P1.y1,P1.z1 = P1.S2[0, ::self.S2_density], P1.S2[1, ::self.S2_density], P1.S2[2, ::self.S2_density]
        values = np.array([P1.x1,P1.y1,P1.z1])
        try:
            kde = stats.gaussian_kde(values)
            P1.d1 = kde(values) #density
            P1.d1 /= P1.d1.max() #relative density, max=1

            splot = mlab.points3d(P1.x1, P1.y1, P1.z1, P1.d1, scale_mode='none',
                                  scale_factor=0.05, figure=Mayavi_S2.fig1)
            cbar = mlab.scalarbar(title='Relative\nDensity\n', orientation='vertical', \
                                  nb_labels=3, label_fmt='%.1f')
        except:
            splot = mlab.points3d(P1.x1, P1.y1, P1.z1, scale_mode='none',
                                  scale_factor=0.05, figure=Mayavi_S2.fig1)

        #####################
        # align-to-grid data:
        phi, theta = np.mgrid[0:np.pi:11j, 0:2*np.pi:11j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        testPlot = mlab.mesh(x, y, z, representation='wireframe', color=(0, 0, 0))
        testPlot.actor.actor.scale = np.array([50,50,50])
        testPlot.actor.property.opacity = 0
        #####################

        splot.actor.actor.scale = np.multiply(self.S2_scale,
                                    np.array([len(P1.df_vol)/float(np.sqrt(2)),
                                              len(P1.df_vol)/float(np.sqrt(2)),
                                              len(P1.df_vol)/float(np.sqrt(2))]))
        
        splot.actor.property.backface_culling = True
        splot.mlab_source.reset

        splot.module_manager.scalar_lut_manager.scalar_bar_widget.repositionable = False
        splot.module_manager.scalar_lut_manager.scalar_bar_widget.resizable = False

        # reposition camera to previous:
        self.scene1.mlab.view(*view)
        self.scene1.mlab.roll(roll)

        def press_callback(vtk_obj, event): #left mouse down callback
            self.click_on = 1

        def hold_callback(vtk_obj, event): #camera rotate callback
            if self.click_on > 0:
                viewS2 = self.scene1.mlab.view(figure=Mayavi_S2.fig1)
                self.phi = '%s%s' % (round(viewS2[0],2), u"\u00b0")
                self.theta = '%s%s' % (round(viewS2[1],2), u"\u00b0")
                #self.scene2.mlab.view(viewS2[0],viewS2[1],viewS2[2],viewS2[3],figure=Mayavi_S2.fig2)

        def release_callback(vtk_obj, event): #left mouse release callback
            if self.click_on == 1:
                self.click_on = 0

        Mayavi_S2.fig1.scene.scene.interactor.add_observer('LeftButtonPressEvent', press_callback)
        Mayavi_S2.fig1.scene.scene.interactor.add_observer('InteractionEvent', hold_callback)
        Mayavi_S2.fig1.scene.scene.interactor.add_observer('EndInteractionEvent', release_callback)
        
        # if display parameters have changed, store them to be grabbed by tab 4:
        MainWindow.S2_scale = self.S2_scale
        MainWindow.S2_iso = self.isosurface_level
        p.S2iso = MainWindow.S2_iso
        p.S2rescale = MainWindow.S2_scale #arcade
        set_params.op(0) #send new GUI data to user parameters file
        
        
    @on_trait_change('isosurface_level') #Electrostatic Potential Map
    def update_scene2(self):
        # store current camera info:
        view = mlab.view()
        roll = mlab.roll()
        
        Mayavi_S2.fig1 = mlab.figure(1, bgcolor=(.5,.5,.5))
        Mayavi_S2.fig2 = mlab.figure(2, bgcolor=(.5,.5,.5))
        
        mlab.sync_camera(Mayavi_S2.fig1, Mayavi_S2.fig2)
        mlab.sync_camera(Mayavi_S2.fig2, Mayavi_S2.fig1)
        
        mlab.clf(figure=Mayavi_S2.fig2)

        if P1.relion_data == True:
            mirror = P1.df_vol[..., ::-1]
            cplot = mlab.contour3d(mirror,contours=self.isosurface_level,
                                   color=(0.9, 0.9, 0.9),
                                   figure=Mayavi_S2.fig2)
            cplot.actor.actor.orientation = np.array([0., -90., 0.])

        else:
            cplot = mlab.contour3d(P1.df_vol,contours=self.isosurface_level,
                                  color=(0.9, 0.9, 0.9), figure=Mayavi_S2.fig2)
        
        cplot.actor.actor.origin = np.array([len(P1.df_vol)/2,len(P1.df_vol)/2,len(P1.df_vol)/2])
        cplot.actor.actor.position = np.array([-len(P1.df_vol)/2,-len(P1.df_vol)/2,-len(P1.df_vol)/2])

        cplot.actor.property.backface_culling = True
        cplot.compute_normals = False
        cplot.mlab_source.reset

        #####################
        # align-to-grid data:
        phi, theta = np.mgrid[0:np.pi:11j, 0:2*np.pi:11j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        testPlot = mlab.mesh(x, y, z, representation='wireframe', color=(0, 0, 0))
        testPlot.actor.actor.scale = np.array([50,50,50])
        testPlot.actor.property.opacity = 0
        ####################

        # reposition camera to previous:
        mlab.view(view[0],view[1],len(P1.df_vol)*2,view[3]) #zoom out based on MRC volume dimensions
        mlab.roll(roll)

        def press_callback(vtk_obj, event): #left mouse down callback
            self.click_on = 1

        def hold_callback(vtk_obj, event): #camera rotate callback
            if self.click_on > 0:
                viewS2 = self.scene2.mlab.view(figure=Mayavi_S2.fig2)
                self.phi = '%s%s' % (round(viewS2[0],2), u"\u00b0")
                self.theta = '%s%s' % (round(viewS2[1],2), u"\u00b0")
                #self.scene1.mlab.view(viewS2[0],viewS2[1],viewS2[2],viewS2[3],figure=Mayavi_S2.fig1)

        def release_callback(vtk_obj, event): #left mouse release callback
            if self.click_on == 1:
                self.click_on = 0

        Mayavi_S2.fig2.scene.scene.interactor.add_observer('LeftButtonPressEvent', press_callback)
        Mayavi_S2.fig2.scene.scene.interactor.add_observer('InteractionEvent', hold_callback)
        Mayavi_S2.fig2.scene.scene.interactor.add_observer('EndInteractionEvent', release_callback)
        
        # if display parameters have changed, store them to be grabbed by tab 4:
        MainWindow.S2_scale = self.S2_scale
        MainWindow.S2_iso = self.isosurface_level
        p.S2iso = MainWindow.S2_iso
        p.S2rescale = MainWindow.S2_scale #arcade
        set_params.op(0) #send new GUI data to user parameters file

    titleLeft = Str
    titleRight = Str

    def _titleLeft_default(self):
        return 'S2 Orientation Distribution'

    def _titleRight_default(self):
        return 'Electrostatic Potential Map'

    @on_trait_change('display_thresh')
    def GCsViewer(self):
        global GCs_window
        try:
            GCs_window.close()
        except:
            pass
        GCs_window = Thresh_Viz()
        GCs_window.setMinimumSize(10, 10)
        GCs_window.setWindowTitle('Projection Direction Thresholding')
        GCs_window.show()


    view = View(VGroup(
            HGroup( #HSplit
                  Group(
                      Item('titleLeft',springy=False,show_label=False,style='readonly',
                           style_sheet='*{font-size:12px; qproperty-alignment:AlignCenter}'),
                      Item('scene1',
                           editor=SceneEditor(scene_class=MayaviScene),
                           height=1, width=1, show_label=False, springy=True,
                           ),
                        ),
                  Group(
                      Item('titleRight',springy=False,show_label=False,style='readonly',
                           style_sheet='*{font-size:12px; qproperty-alignment:AlignCenter}'),
                      Item('scene2',
                           editor=SceneEditor(scene_class=MayaviScene),
                           height=1, width=1, show_label=False, springy=True,
                           ),
                      ),
                  ),
            HGroup(
                HGroup(
                    Item('display_thresh',springy=True,show_label=False,
                         tooltip='Display the occupancy of each PD.'),
                    Item('S2_scale',springy=True,show_label=True,
                        tooltip='Change the relative scale of S2 with respect to the volume map above.'),
                    Item('S2_density',springy=True,show_label=True,
                         tooltip='Density of available points displayed on S2.'),
                    show_border=True,orientation='horizontal'),
                HGroup(
                    Item('phi',springy=True,show_label=True,#style='readonly',
                         editor=TextEditor(evaluate=float),
                         enabled_when='phi == float(0)', #i.e., never
                         ),
                    #Item('_'),
                    Item('theta',springy=True,show_label=True,#style='readonly',
                         editor=TextEditor(evaluate=float),
                         enabled_when='phi == float(0)', #i.e., never
                         ),
                    Item('isosurface_level',springy=True,show_label=True,
                         tooltip='Change the isosurface level of the volume map above.'),
                    show_border=True,orientation='horizontal'),
                ),
                ),
                resizable=True,
                )

################################################################################
# mayaVI viz 2 (Conformational Coordinates):
    
class Mayavi_Rho(HasTraits): #electrostatic potential map, P4
    scene3 = Instance(MlabSceneModel, ())
    PrD_high = 2
    isosurface = Range(2,9,3, mode='enum')
    volume_alpha = Enum(1.0,.8,.6,.4,.2,0.0)
    S2_scale = Float
    anchorsUpdate = []
    trashUpdate = []
    phi = Str
    theta = Str
    click_on = 0
    click_on_Eul = 0

    def _phi_default(self):
        return '%s%s' % (0, u"\u00b0")

    def _theta_default(self):
        return '%s%s' % (0, u"\u00b0")
    
    def _S2_scale_default(self):
        return float(1)
    
    def update_S2(self):
        self.S2_scale = float(MainWindow.S2_scale)
        self.isosurface = int(MainWindow.S2_iso)

    def view_anglesP4(self, dialog):
        viewRho = self.scene3.mlab.view(figure=Mayavi_Rho.fig3)
        azimuth = viewRho[0] #phi: 0-360
        elevation = viewRho[1] #theta: 0-180
        zoom = viewRho[2]
        if dialog is True:
            print_anglesP4(azimuth, elevation)
        else:
            return zoom
            
    def update_viewP4(self, azimuth, elevation, distance):        
        self.scene3.mlab.view(azimuth=azimuth,
                              elevation=elevation,
                              distance=distance,
                              reset_roll=False,
                              figure=Mayavi_Rho.fig3)

    def update_euler(self):
        self.phi = '%s%s' % (round(float(P3.phi[(P4.user_PrD)-1]), 2), u"\u00b0")
        self.theta = '%s%s' % (round(float(P3.theta[(P4.user_PrD)-1]), 2), u"\u00b0")
                              
    @on_trait_change('S2_scale,volume_alpha,isosurface')
    def update_scene3(self):        
        # store current camera info:
        view = self.scene3.mlab.view()
        roll = self.scene3.mlab.roll()
      
        Mayavi_Rho.fig3 = mlab.figure(3)
        self.scene3.background = (0.0, 0.0, 0.0)

        if P3.dictGen == False: #if P3 dictionaries not yet created (happens once on P3)
            mlab.clf(figure=Mayavi_Rho.fig3)
            
            # =================================================================
            # Volume (contour):
            # =================================================================
            mirror = P1.df_vol[..., ::-1]
            cplot = mlab.contour3d(mirror,contours=self.isosurface,
                                   color=(0.5, 0.5, 0.5),
                                   figure=Mayavi_Rho.fig3)
            cplot.actor.actor.orientation = np.array([0., -90., 0.]) #ZULU?

            cplot.actor.actor.origin = np.array([len(P1.df_vol)/2,len(P1.df_vol)/2,len(P1.df_vol)/2])
            cplot.actor.actor.position = np.array([-len(P1.df_vol)/2,-len(P1.df_vol)/2,-len(P1.df_vol)/2])

            cplot.actor.property.backface_culling = True
            cplot.compute_normals = False
            cplot.actor.property.opacity = self.volume_alpha

            # =================================================================
            # Align-to-grid data:
            # =================================================================
            phi, theta = np.mgrid[0:np.pi:11j, 0:2*np.pi:11j]
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            testPlot = mlab.mesh(x, y, z, representation='wireframe', color=(1, 1, 1))
            testPlot.actor.actor.scale = np.multiply(self.S2_scale,
                                                np.array([len(P1.df_vol)/float(np.sqrt(2)),
                                                          len(P1.df_vol)/float(np.sqrt(2)),
                                                          len(P1.df_vol)/float(np.sqrt(2))]))
            testPlot.actor.property.opacity = 0

            # =================================================================
            # S2 Distribution (scatter):
            # =================================================================
            splot = mlab.points3d(P1.x2,P1.y2,P1.z2,P3.col, scale_mode='none',
                                  scale_factor=0.05, figure=Mayavi_Rho.fig3)
            splot.actor.property.backface_culling = True

            splot.actor.actor.scale = np.multiply(self.S2_scale,
                                        np.array([len(P1.df_vol)/float(np.sqrt(2)),
                                                  len(P1.df_vol)/float(np.sqrt(2)),
                                                  len(P1.df_vol)/float(np.sqrt(2))]))
            splot.actor.actor.origin = np.array([0,0,0])
            splot.actor.actor.position = np.array([0,0,0])

            # =================================================================
            # S2 Anchors (sparse scatter):
            # =================================================================
            aplot = mlab.points3d(P1.x3,P1.y3,P1.z3, scale_mode='none',
                                  scale_factor=0.06, figure=Mayavi_Rho.fig3)

            aplot.actor.property.backface_culling = True
            aplot.glyph.color_mode = 'no_coloring'
            aplot.actor.property.color = (1.0, 1.0, 1.0)
            aplot.actor.actor.scale = np.multiply(self.S2_scale,
                                        np.array([len(P1.df_vol)/float(np.sqrt(2)),
                                                  len(P1.df_vol)/float(np.sqrt(2)),
                                                  len(P1.df_vol)/float(np.sqrt(2))]))
            aplot.actor.actor.origin = np.array([0,0,0])
            aplot.actor.actor.position = np.array([0,0,0])

            self.anchorsUpdate = aplot.mlab_source

            # =================================================================
            # S2 Trash (sparse scatter):
            # =================================================================
            tplot = mlab.points3d(P1.x4,P1.y4,P1.z4, scale_mode='none',
                                  scale_factor=0.06, figure=Mayavi_Rho.fig3)

            tplot.actor.property.backface_culling = True
            tplot.glyph.color_mode = 'no_coloring'
            tplot.actor.property.color = (0.0, 0.0, 0.0)
            tplot.actor.actor.scale = np.multiply(self.S2_scale,
                                        np.array([len(P1.df_vol)/float(np.sqrt(2)),
                                                  len(P1.df_vol)/float(np.sqrt(2)),
                                                  len(P1.df_vol)/float(np.sqrt(2))]))
            tplot.actor.actor.origin = np.array([0,0,0])
            tplot.actor.actor.position = np.array([0,0,0])

            self.trashUpdate = tplot.mlab_source

        else: #only update anchors
            self.anchorsUpdate.reset(x=P1.x3, y=P1.y3, z=P1.z3)
            self.trashUpdate.reset(x=P1.x4, y=P1.y4, z=P1.z4)

        # =====================================================================
        # reposition camera to previous:
        # =====================================================================
        mlab.view(*view)
        mlab.roll(roll)
        
        def press_callback(vtk_obj, event): #left mouse down callback
            self.click_on = 1

        def release_callback(vtk_obj, event): #left mouse release callback
            if self.click_on == 1:
                self.click_on = 0
                # =============================================================
                # magnetize to nearest PrD:
                # =============================================================
                # CONVENTIONS:
                # =============================================================
                # mayavi angle 0 -> [-180, 180]: azimuth, phi, longitude
                # mayavi angle 1 -> [0, 180]: elevation/inclination, theta, latitude
                # =============================================================
                angles = self.scene3.mlab.view(figure=Mayavi_Rho.fig3)
                phi0 = angles[0]*np.pi/180
                theta0 = angles[1]*np.pi/180

                r0 = 1
                x0 = (r0*np.sin(theta0)*np.cos(phi0))
                y0 = (r0*np.sin(theta0)*np.sin(phi0))
                z0 = (r0*np.cos(theta0))
                
                # read points from tesselated sphere:
                fname = os.path.join(P1.user_directory,'outputs_{}/topos/Euler_PrD/PrD_map.txt'.format(p.proj_name))
                data = []
                with open(fname) as values:
                    for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                        data.append(column)

                prds = data[0]
                thetas = data[1]
                phis = data[2]
                psis = data[3]
                xs = data[4]
                ys = data[5]
                zs = data[6]

                xyz = np.column_stack((xs,ys,zs))
                angs = np.column_stack((thetas,phis))
                dists = [] #distances between current and all tesselated points
                indexes = []
                idx = 0
                # find nearest neighbor (pythagorean):
                for i,j,k in xyz:
                    x1 = float(i)
                    y1 = float(j)
                    z1 = float(k)
                    
                    d = np.sqrt( (x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2 )

                    if not dists:
                        dists.append(d)
                        indexes.append(idx)
                    if dists:
                        if d < np.amin(dists):
                            dists.append(d)
                            indexes.append(idx)
                    idx += 1
                # update view:
                current = self.scene3.mlab.view(figure=Mayavi_Rho.fig3) #grab current distance
                self.scene3.mlab.view(azimuth=float(phis[indexes[-1]]),
                                      elevation=float(thetas[indexes[-1]]),
                                      distance=current[2],
                                      roll=float(phis[indexes[-1]])+270,
                                      #reset_roll=False, #ZULU: still need to correct volume in-plane rotation to match image's
                                      figure=Mayavi_Rho.fig3)
                P4.entry_PrD.setValue(indexes[-1]+1) #update PrD and thus topos
                self.phi = '%s%s' % (round(float(phis[indexes[-1]]),2), u"\u00b0")
                self.theta = '%s%s' % (round(float(thetas[indexes[-1]]),2), u"\u00b0")
                                      
        Mayavi_Rho.fig3.scene.scene.interactor.add_observer('LeftButtonPressEvent', press_callback)
        Mayavi_Rho.fig3.scene.scene.interactor.add_observer('EndInteractionEvent', release_callback)

        # live update of Euler angles:
        def press_callback_Eul(vtk_obj, event): #left mouse down callback
            self.click_on_Eul = 1

        def hold_callback_Eul(vtk_obj, event): #camera rotate callback
            if self.click_on_Eul > 0:
                viewS2 = self.scene3.mlab.view(figure=Mayavi_Rho.fig3)
                self.phi = '%s%s' % (round(viewS2[0],2), u"\u00b0")
                self.theta = '%s%s' % (round(viewS2[1],2), u"\u00b0")

        def release_callback_Eul(vtk_obj, event): #left mouse release callback
            if self.click_on_Eul == 1:
                self.click_on_Eul = 0

        Mayavi_Rho.fig3.scene.scene.interactor.add_observer('LeftButtonPressEvent', press_callback_Eul)
        Mayavi_Rho.fig3.scene.scene.interactor.add_observer('InteractionEvent', hold_callback_Eul)
        Mayavi_Rho.fig3.scene.scene.interactor.add_observer('EndInteractionEvent', release_callback_Eul)


    title = Str
    def _title_default(self):
        return 'Electrostatic Potential Map'
        
    view = View(VGroup(
                    Group(
                        Item('title',springy=False,show_label=False,style='readonly',
                             style_sheet='*{font: "Arial"; font-size:12px; qproperty-alignment:AlignCenter}'),
                        Item('scene3', editor = SceneEditor(scene_class=MayaviScene),
                            height=1, width=1, show_label=False, springy=True),
                        ),
                VGroup(
                    HGroup(
                        Item('phi',springy=True,show_label=True,#style='readonly',
                             editor=TextEditor(evaluate=float),
                             enabled_when='phi == float(0)', #i.e., never
                             ),
                        Item('theta',springy=True,show_label=True,#style='readonly',
                             editor=TextEditor(evaluate=float),
                             enabled_when='phi == float(0)', #i.e., never
                             ),
                    ),
                    show_border=False, orientation='vertical'
                ),
                ),
                resizable=True,
                )
    
# =============================================================================
# GUI tab 1:
# =============================================================================

class P1(QtGui.QWidget):
    # user inputs:
    user_volume = ''
    user_mask = ''
    df_vol = ''
    user_stack = ''
    user_alignment = ''
    user_name = time_init
    user_directory = pyDir
    # full S2 angles:
    x1 = []
    y1 = []
    z1 = []
    d1 = []
    # thresholded S2 angles:
    x2 = []
    y2 = []
    z2 = []
    # S2 anchor nodes:
    x3 = []
    y3 = []
    z3 = []
    a3 = []
    # S2 trash nodes:
    x4 = []
    y4 = []
    z4 = []
    a4 = []
    # remaining parameters:
    user_pixel = 0.01
    user_diameter = 0.01
    user_resolution = 0.01
    user_aperture = 1
    user_shannon = 0.0
    user_width = 0.0
    S2rescale = 1.0
    relion_data = True #True for .star (ZULU: need to update this to instead correspond to different RELION versions; 'False' no longer used)
    q = []
    S2 = []
    df = []
    CG = []
    shifts = []
    all_PrDs = []
    all_occ = []
    all_phi = []
    all_theta = []
    all_psi = []
    S2_density_all = [5,10,25,50,100,250,500,1000,10000,100000] #needs to match Mayavi_S2.S2_density_all!

    def __init__(self, parent=None):
        super(P1, self).__init__(parent)
        layout = QtGui.QGridLayout(self)
        layout.setContentsMargins(20,20,20,20) #W,N,E,S
        layout.setSpacing(10)        
                
        # average volume input:        
        def choose_avgVol():
            fileName = QtGui.QFileDialog.getOpenFileName(self, 'Choose Data File', '',
                                                         ('Data Files (*.mrc)'))[0]
                                                        #('MRC (*.mrc);;SPI (*.spi)'))
                                                        #('All files (*.*);;
            if fileName:
                P1.entry_avgVol.setDisabled(False)
                P1.entry_avgVol.setText(fileName)
                P1.entry_avgVol.setDisabled(True)
                # change volume data:
                P1.user_volume = fileName

                with mrcfile.open(P1.user_volume, mode='r+') as mrc:
                    mrc.header.mapc = 1
                    mrc.header.mapr = 2
                    mrc.header.maps = 3
                    P1.df_vol = mrc.data
            else:
                P1.entry_avgVol.setDisabled(False)
                P1.entry_avgVol.setText('Filename')
                P1.entry_avgVol.setDisabled(True)
                P1.user_volume = ''
            p.avg_vol_file = P1.user_volume
              
        self.label_edge1 = QtGui.QLabel('')
        self.label_edge1.setMargin(0)
        self.label_edge1.setLineWidth(1)
        self.label_edge1.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge1, 0, 0, 1, 6)
        self.label_edge1.show()

        self.label_edge1a = QtGui.QLabel('')
        self.label_edge1a.setMargin(0)
        self.label_edge1a.setLineWidth(1)
        self.label_edge1a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge1a, 0, 0, 1, 1)
        self.label_edge1a.show()

        self.label_avgVol = QtGui.QLabel('Average Volume')
        self.label_avgVol.setFont(font_standard)
        self.label_avgVol.setMargin(20)
        self.label_avgVol.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_avgVol, 0, 0, 1, 1)
        self.label_avgVol.show()

        P1.entry_avgVol = QtGui.QLineEdit('Filename')
        P1.entry_avgVol.setDisabled(True)
        layout.addWidget(P1.entry_avgVol, 0, 1, 1, 4)
        P1.entry_avgVol.show()

        self.button_browse1 = QtGui.QPushButton('          Browse          ', self)
        self.button_browse1.clicked.connect(choose_avgVol)
        self.button_browse1.setToolTip('Accepted format: <i>.mrc</i>')
        layout.addWidget(self.button_browse1, 0, 5, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.button_browse1.show()

        # alignment input:
        def choose_align():
            fileName = QtGui.QFileDialog.getOpenFileName(self, 'Choose Data File', '',
                                                         ('Data Files (*.star)'))[0]
                                                        #('All files (*.*)'))
            if fileName:
                if fileName.endswith('.star'):
                    P1.relion_data = True
                else:
                    P1.relion_data = False
                p.relion_data = P1.relion_data
                P1.entry_align.setDisabled(False)
                P1.entry_align.setText(fileName)
                P1.entry_align.setDisabled(True)
                P1.user_alignment = fileName
            else:
                P1.entry_align.setDisabled(False)
                P1.entry_align.setText('Filename')
                P1.entry_align.setDisabled(True)
                P1.user_alignment = ''
            p.align_param_file = P1.user_alignment

        self.label_edge3 = QtGui.QLabel('')
        self.label_edge3.setMargin(20)
        self.label_edge3.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge3, 1, 0, 1, 6)
        self.label_edge3.show()

        self.label_edge3a = QtGui.QLabel('')
        self.label_edge3a.setMargin(20)
        self.label_edge3a.setLineWidth(1)
        self.label_edge3a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge3a, 1, 0, 1, 1)
        self.label_edge3a.show()

        self.label_align = QtGui.QLabel('Alignment File')
        self.label_align.setFont(font_standard)
        self.label_align.setMargin(20)
        self.label_align.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_align, 1, 0, 1, 1)
        self.label_align.show()

        P1.entry_align = QtGui.QLineEdit('Filename')
        P1.entry_align.setDisabled(True)
        layout.addWidget(P1.entry_align, 1, 1, 1, 4)
        P1.entry_align.show()

        self.button_browse3 = QtGui.QPushButton('          Browse          ', self)
        self.button_browse3.clicked.connect(choose_align)
        self.button_browse3.setToolTip('Accepted format: <i>.star</i>')
        layout.addWidget(self.button_browse3, 1, 5, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.button_browse3.show()

        # image stack input:
        def choose_imgStack():
            fileName = QtGui.QFileDialog.getOpenFileName(self, 'Choose Data File', '',
                                                         ('Data Files (*.mrcs)'))[0]
            if fileName:
                P1.entry_imgStack.setDisabled(False)
                P1.entry_imgStack.setText(fileName)
                P1.entry_imgStack.setDisabled(True)
                P1.user_stack = fileName
                if fileName.endswith('.mrcs'): #relion file format
                    P1.relion_data = True
                    mrc = mrcfile.mmap(fileName, mode='r+')
                    mrc.set_image_stack()
                    p.nPix = np.shape(mrc.data[0])[0]
            else:
                P1.entry_imgStack.setDisabled(False)
                P1.entry_imgStack.setText('Filename')
                P1.entry_imgStack.setDisabled(True)
                P1.user_stack = ''
            p.img_stack_file = P1.user_stack

        self.label_edge2 = QtGui.QLabel('')
        self.label_edge2.setMargin(20)
        self.label_edge2.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge2, 2, 0, 1, 6)
        self.label_edge2.show()

        self.label_edge2a = QtGui.QLabel('')
        self.label_edge2a.setMargin(20)
        self.label_edge2a.setLineWidth(1)
        self.label_edge2a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge2a, 2, 0, 1, 1)
        self.label_edge2a.show()
        
        self.label_imgStack = QtGui.QLabel('Image Stack')
        self.label_imgStack.setFont(font_standard)
        self.label_imgStack.setMargin(5)
        self.label_imgStack.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_imgStack, 2, 0, 1, 1)
        self.label_imgStack.show()

        P1.entry_imgStack = QtGui.QLineEdit('Filename')
        P1.entry_imgStack.setDisabled(True)
        layout.addWidget(P1.entry_imgStack, 2, 1, 1, 4)
        P1.entry_imgStack.show()

        self.button_browse2 = QtGui.QPushButton('          Browse          ', self)
        self.button_browse2.clicked.connect(choose_imgStack)
        self.button_browse2.setToolTip('Accepted format: <i>.mrcs</i>')
        layout.addWidget(self.button_browse2, 2, 5, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.button_browse2.show()

        # mask volume input:
        def choose_maskVol():
            fileName = QtGui.QFileDialog.getOpenFileName(self, 'Choose Data File', '',
                                                         ('Data Files (*.mrc)'))[0]
            if fileName:
                P1.entry_maskVol.setDisabled(False)
                P1.entry_maskVol.setText(fileName)
                P1.entry_maskVol.setDisabled(True)
                # change volume data:
                P1.user_mask = fileName
            else:
                P1.entry_maskVol.setDisabled(False)
                P1.entry_maskVol.setText('Filename')
                P1.entry_maskVol.setDisabled(True)
                P1.user_mask = ''
            p.mask_vol_file = P1.user_mask

        self.label_edgeM = QtGui.QLabel('')
        self.label_edgeM.setMargin(0)
        self.label_edgeM.setLineWidth(1)
        self.label_edgeM.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edgeM, 3, 0, 1, 6)
        self.label_edgeM.show()

        self.label_edgeMa = QtGui.QLabel('')
        self.label_edgeMa.setMargin(0)
        self.label_edgeMa.setLineWidth(1)
        self.label_edgeMa.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edgeMa, 3, 0, 1, 1)
        self.label_edgeMa.show()

        self.label_maskVol = QtGui.QLabel('Mask Volume')
        self.label_maskVol.setFont(font_standard)
        self.label_maskVol.setMargin(20)
        self.label_maskVol.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_maskVol, 3, 0, 1, 1)
        self.label_maskVol.show()

        P1.entry_maskVol = QtGui.QLineEdit('Filename')
        P1.entry_maskVol.setDisabled(True)
        layout.addWidget(P1.entry_maskVol, 3, 1, 1, 4)
        P1.entry_maskVol.show()

        self.button_browseM = QtGui.QPushButton('          Browse          ', self)
        self.button_browseM.clicked.connect(choose_maskVol)
        self.button_browseM.setToolTip('Accepted format: <i>.mrc</i>')
        layout.addWidget(self.button_browseM, 3, 5, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.button_browseM.show()

        def choose_name():
            text, ok = QtGui.QInputDialog.getText(self, 'ManifoldEM Project Name', 'Enter project name:')

            if ok:
                if (0 < len(text) <= 30) and (re.match('^[A-Za-z0-9-_]*$', text) != None):
                    P1.user_name = text
                    P1.entry_name.setDisabled(False)
                    P1.entry_name.setText(text)
                    P1.entry_name.setDisabled(True)
                else:
                    box = QtGui.QMessageBox(self)
                    box.setWindowTitle('%s Error' % progname)
                    box.setText('<b>Input Error</b>')
                    box.setFont(font_standard)
                    box.setIcon(QtGui.QMessageBox.Information)
                    box.setInformativeText('Project names must be no greater than 20 characters in length,\
                                            and can only contain members from the following set:\
                                            <br />\
                                            {A-Z, a-z, 0-9, -, _ }\
                                            <br /><br />\
                                            Please choose a project name that matches these criteria.')
                    box.setStandardButtons(QtGui.QMessageBox.Ok)
                    box.setDefaultButton(QtGui.QMessageBox.Ok)
                    ret = box.exec_()

                    P1.user_name = time_init
                    P1.entry_name.setDisabled(False)
                    P1.entry_name.setText(time_init)
                    P1.entry_name.setDisabled(True)
            else:
                P1.user_name = time_init
                P1.entry_name.setDisabled(False)
                P1.entry_name.setText(time_init)
                P1.entry_name.setDisabled(True)
            p.proj_name = P1.user_name

        self.label_edge6 = QtGui.QLabel('')
        self.label_edge6.setMargin(20)
        self.label_edge6.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge6, 4, 0, 1, 6)
        self.label_edge6.show()

        self.label_edge6a = QtGui.QLabel('')
        self.label_edge6a.setMargin(20)
        self.label_edge6a.setLineWidth(1)
        self.label_edge6a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge6a, 4, 0, 1, 1)
        self.label_edge6a.show()

        self.label_dir = QtGui.QLabel('Project Name')
        self.label_dir.setFont(font_standard)
        self.label_dir.setMargin(20)
        self.label_dir.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_dir, 4, 0, 1, 1)
        self.label_dir.show()

        P1.entry_name = QtGui.QLineEdit(time_init)
        P1.entry_name.setDisabled(True)
        layout.addWidget(P1.entry_name, 4, 1, 1, 4)
        P1.entry_name.show()

        self.button_browse4 = QtGui.QPushButton('          Choose          ', self)
        self.button_browse4.clicked.connect(choose_name)
        self.button_browse4.setToolTip('Choose project name.')
        layout.addWidget(self.button_browse4, 4, 5, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.button_browse4.show()

        def calc_shannon():
            P1.user_shannon = np.divide(P1.user_resolution, P1.user_diameter)
            P1.entry_shannon.setValue(P1.user_shannon)
            p.sh = P1.user_shannon

        def calc_angWidth():
            P1.user_width = ((P1.user_aperture)*(P1.user_resolution)) / P1.user_diameter

            if P1.user_width > np.sqrt(4*np.pi):
                P1.user_width = np.sqrt(4*np.pi) #max value constraint

            P1.entry_angWidth.setValue(P1.user_width)
            p.ang_width = P1.user_width

        # pixel size input:
        def choose_pixel():
            P1.user_pixel = P1.entry_pixel.value()
            p.pix_size = P1.user_pixel

        self.label_edge4 = QtGui.QLabel('')
        self.label_edge4.setMargin(20)
        self.label_edge4.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge4, 5, 0, 1, 2)
        self.label_edge4.show()

        self.label_edge4a = QtGui.QLabel('')
        self.label_edge4a.setMargin(20)
        self.label_edge4a.setLineWidth(1)
        self.label_edge4a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge4a, 5, 0, 1, 1)
        self.label_edge4a.show()

        self.label_pixel = QtGui.QLabel('Pixel Size')
        self.label_pixel.setFont(font_standard)
        self.label_pixel.setMargin(20)
        self.label_pixel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_pixel, 5, 0, 1, 1)
        self.label_pixel.show()

        P1.entry_pixel = QtGui.QDoubleSpinBox(self)
        P1.entry_pixel.setMinimum(0.001)
        P1.entry_pixel.setMaximum(1000.00)
        P1.entry_pixel.setDecimals(3)
        P1.entry_pixel.valueChanged.connect(choose_pixel)
        P1.entry_pixel.setSuffix(' %s' % (u"\u00c5"))
        P1.entry_pixel.setStyleSheet("QDoubleSpinBox { width : 150px }")
        layout.addWidget(P1.entry_pixel, 5, 1, 1, 1, QtCore.Qt.AlignLeft)
        P1.entry_pixel.show()

        # object diameter input:
        def choose_diameter():
            P1.user_diameter = P1.entry_objDiam.value()
            p.obj_diam = P1.user_diameter

        self.label_edge5 = QtGui.QLabel('')
        self.label_edge5.setMargin(20)
        self.label_edge5.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge5, 5, 2, 1, 2)
        self.label_edge5.show()

        self.label_edge5a = QtGui.QLabel('')
        self.label_edge5a.setMargin(20)
        self.label_edge5a.setLineWidth(1)
        self.label_edge5a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge5a, 5, 2, 1, 1)
        self.label_edge5a.show()

        self.label_objDiam = QtGui.QLabel('Object Diameter')
        self.label_objDiam.setFont(font_standard)
        self.label_objDiam.setMargin(20)
        self.label_objDiam.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_objDiam, 5, 2, 1, 1)
        self.label_objDiam.show()

        P1.entry_objDiam = QtGui.QDoubleSpinBox(self)
        P1.entry_objDiam.setMinimum(0.01)
        P1.entry_objDiam.setMaximum(10000.00)
        P1.entry_objDiam.setSuffix(' %s' % (u"\u00c5"))
        P1.entry_objDiam.valueChanged.connect(choose_diameter)
        P1.entry_objDiam.valueChanged.connect(calc_shannon)
        P1.entry_objDiam.valueChanged.connect(calc_angWidth)
        P1.entry_objDiam.setStyleSheet("QDoubleSpinBox { width : 150px }")
        layout.addWidget(P1.entry_objDiam, 5, 3, 1, 1, QtCore.Qt.AlignLeft)
        P1.entry_objDiam.show()

        # shannon angle output:
        self.label_edgeSh = QtGui.QLabel('')
        self.label_edgeSh.setMargin(20)
        self.label_edgeSh.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edgeSh, 5, 4, 1, 2)
        self.label_edgeSh.show()

        self.label_edgeSha = QtGui.QLabel('')
        self.label_edgeSha.setMargin(20)
        self.label_edgeSha.setLineWidth(1)
        self.label_edgeSha.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edgeSha, 5, 4, 1, 1)
        self.label_edgeSha.show()

        self.label_shannon = QtGui.QLabel('Shannon Angle')
        self.label_shannon.setFont(font_standard)
        self.label_shannon.setMargin(20)
        self.label_shannon.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_shannon, 5, 4, 1, 1)
        self.label_shannon.show()

        P1.entry_shannon = QtGui.QDoubleSpinBox(self)
        P1.entry_shannon.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)
        P1.entry_shannon.setDecimals(4)
        P1.entry_shannon.setSuffix(' rad')
        P1.entry_shannon.setDisabled(True)
        P1.entry_shannon.setStyleSheet("QDoubleSpinBox { width : 150px }")
        layout.addWidget(P1.entry_shannon, 5, 5, 1, 1, QtCore.Qt.AlignLeft)
        P1.entry_shannon.show()

        # resolution estimate input:
        def choose_resolution():
            P1.user_resolution = P1.entry_resolution.value()
            p.resol_est = P1.user_resolution

        self.label_edge7 = QtGui.QLabel('')
        self.label_edge7.setMargin(20)
        self.label_edge7.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge7, 6, 0, 1, 2)
        self.label_edge7.show()

        self.label_edge7a = QtGui.QLabel('')
        self.label_edge7a.setMargin(20)
        self.label_edge7a.setLineWidth(1)
        self.label_edge7a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge7a, 6, 0, 1, 1)
        self.label_edge7a.show()

        self.label_resolution = QtGui.QLabel('Resolution')
        self.label_resolution.setFont(font_standard)
        self.label_resolution.setMargin(5)
        self.label_resolution.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_resolution, 6, 0, 1, 1)
        self.label_resolution.show()

        P1.entry_resolution = QtGui.QDoubleSpinBox(self)
        P1.entry_resolution.setMinimum(0.01)
        P1.entry_resolution.setMaximum(1000.00)
        P1.entry_resolution.setDecimals(2)
        P1.entry_resolution.setSuffix(' %s' % (u"\u00c5"))
        P1.entry_resolution.valueChanged.connect(choose_resolution)
        P1.entry_resolution.valueChanged.connect(calc_shannon)
        P1.entry_resolution.valueChanged.connect(calc_angWidth)
        P1.entry_resolution.setStyleSheet("QDoubleSpinBox { width : 150px }")
        layout.addWidget(P1.entry_resolution, 6, 1, 1, 1, QtCore.Qt.AlignLeft)
        P1.entry_resolution.show()

        # aperture index input:
        def choose_aperture():
            P1.user_aperture = int(P1.entry_aperture.value())
            p.ap_index = P1.user_aperture
        
        self.label_edge8 = QtGui.QLabel('')
        self.label_edge8.setMargin(20)
        self.label_edge8.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge8, 6, 2, 1, 2)
        self.label_edge8.show()

        self.label_edge8a = QtGui.QLabel('')
        self.label_edge8a.setMargin(20)
        self.label_edge8a.setLineWidth(1)
        self.label_edge8a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge8a, 6, 2, 1, 1)
        self.label_edge8a.show()

        self.label_aperture = QtGui.QLabel('Aperture Index')
        self.label_aperture.setFont(font_standard)
        self.label_aperture.setMargin(20)
        self.label_aperture.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_aperture, 6, 2, 1, 1)
        self.label_aperture.show()

        P1.entry_aperture = QtGui.QDoubleSpinBox(self)
        P1.entry_aperture.setMinimum(1)
        P1.entry_aperture.setMaximum(1000)
        P1.entry_aperture.setDecimals(0)
        P1.entry_aperture.valueChanged.connect(choose_aperture)
        P1.entry_aperture.valueChanged.connect(calc_angWidth)
        P1.entry_aperture.setStyleSheet("QDoubleSpinBox { width : 150px }")
        layout.addWidget(P1.entry_aperture, 6, 3, 1, 1, QtCore.Qt.AlignLeft)
        P1.entry_aperture.show()

        # angle width output:
        self.label_edgeAw1 = QtGui.QLabel('')
        self.label_edgeAw1.setMargin(20)
        self.label_edgeAw1.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edgeAw1, 6, 4, 1, 2)
        self.label_edgeAw1.show()

        self.label_edgeAw2 = QtGui.QLabel('')
        self.label_edgeAw2.setMargin(20)
        self.label_edgeAw2.setLineWidth(1)
        self.label_edgeAw2.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edgeAw2, 6, 4, 1, 1)
        self.label_edgeAw2.show()

        self.label_angWidth = QtGui.QLabel('Angle Width')
        self.label_angWidth.setFont(font_standard)
        self.label_angWidth.setMargin(20)
        self.label_angWidth.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_angWidth, 6, 4, 1, 1)
        self.label_angWidth.show()

        P1.entry_angWidth = QtGui.QDoubleSpinBox(self)
        P1.entry_angWidth.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)
        P1.entry_angWidth.setDecimals(4)
        P1.entry_angWidth.setMaximum(np.sqrt(4*np.pi))
        P1.entry_angWidth.setSuffix(' rad')
        P1.entry_angWidth.setDisabled(True)
        P1.entry_angWidth.setStyleSheet("QDoubleSpinBox { width : 150px }")
        layout.addWidget(P1.entry_angWidth, 6, 5, 1, 1, QtCore.Qt.AlignLeft)
        P1.entry_angWidth.show()
    
        # next page:
        self.label_Hline = QtGui.QLabel("") #aesthetic line left
        self.label_Hline.setFont(font_standard)
        self.label_Hline.setMargin(0)
        self.label_Hline.setFrameStyle(QtGui.QFrame.HLine | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_Hline, 7, 0, 1, 2, QtCore.Qt.AlignVCenter)
        self.label_Hline.show()
        
        self.label_Hline = QtGui.QLabel("") #aesthetic line right
        self.label_Hline.setFont(font_standard)
        self.label_Hline.setMargin(0)
        self.label_Hline.setFrameStyle(QtGui.QFrame.HLine | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_Hline, 7, 4, 1, 2, QtCore.Qt.AlignVCenter)
        self.label_Hline.show()

        P1.button_toP2 = QtGui.QPushButton('View Orientation Distribution', self)
        P1.button_toP2.setToolTip('All entries must be complete.')
        layout.addWidget(P1.button_toP2, 7, 2, 1, 2)
        P1.button_toP2.show()

# =============================================================================
# GUI tab 2:
# =============================================================================

class P2(QtGui.QWidget):
    # user inputs:
    user_azimuth = 0
    user_elevation = 0
    noReturnFinal = False
    
    # page 2 layout:    
    def __init__(self, parent=None):
        super(P2, self).__init__(parent)
        layout = QtGui.QGridLayout(self)
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(10)

        global print_anglesP2
        def print_anglesP2(azimuth,elevation):
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s' % progname)
            box.setText('<b>Current Angles</b>')
            box.setFont(font_standard)
            box.setInformativeText('Azimuth (phi): %s%s<br /><br />Elevation (theta): %s%s< br />'
                                   % (round(azimuth,2), u"\u00B0", round(elevation,2), u"\u00B0"))
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setDefaultButton(QtGui.QMessageBox.Ok)
            ret = box.exec_()

        P2.viz1 = Mayavi_S2()
        P2.ui1 = P2.viz1.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(P2.ui1, 0, 0, 1, 6)

        # next page:                            
        self.label_Hline = QtGui.QLabel("") #aesthetic line left
        self.label_Hline.setFont(font_standard)
        self.label_Hline.setMargin(20)
        self.label_Hline.setFrameStyle(QtGui.QFrame.HLine | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_Hline, 2, 0, 1, 2, QtCore.Qt.AlignVCenter)
        self.label_Hline.show()
        
        self.label_Hline = QtGui.QLabel("") #aesthetic line right
        self.label_Hline.setFont(font_standard)
        self.label_Hline.setMargin(20)
        self.label_Hline.setFrameStyle(QtGui.QFrame.HLine | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_Hline, 2, 4, 1, 2, QtCore.Qt.AlignVCenter)
        self.label_Hline.show()
        
        self.button_binPart = QtGui.QPushButton('Bin Particles')
        self.button_binPart.setToolTip('Proceed to embedding.')
        layout.addWidget(self.button_binPart, 2, 2, 1, 2)
        self.button_binPart.show()
        

# =============================================================================
# GUI tab 3:
# =============================================================================

class P3(QtGui.QWidget):
    # temporary values:
    PrD_total = 2
    theta = 0
    phi = 0
    col = []
    user_processors = 1
    user_psi = 5
    user_dimensions = 1
    dictGen = False
    # threading:
    progress1Changed = QtCore.Signal(int)
    progress2Changed = QtCore.Signal(int)
    progress3Changed = QtCore.Signal(int)
    progress4Changed = QtCore.Signal(int)

    def __init__(self, parent=None):
        super(P3, self).__init__(parent)
        layout = QtGui.QGridLayout(self)
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(10)

        def choose_processors():
            P3.user_processors = P3.entry_proc.value()
            p.ncpu = P3.entry_proc.value()
            P5.entry_proc.setValue(P3.entry_proc.value())

        def choose_psi():
            P3.user_psi = P3.entry_psi.value()
            p.num_psis = P3.entry_psi.value()

        def choose_dimensions():
            dim_temp = P3.user_dimensions
            if P4.recompute == 1: #overwrite warning
                msg = 'The energy landscape has been previously computed under a different\
                        dimensionality. To recompute final outputs under a new dimensionality,\
                        all data rendered after the <i>Eigenvectors</i> tab must be overwritten.\
                        If you choose to pursue this option, make sure to save your current \
                        anchors and removals to file (on the next tab) before exiting the program,\
                        since this operation will revert that progress.\
                        <br /><br />\
                        Do you want to proceed?'
                box = QtGui.QMessageBox(self)
                box.setWindowTitle('%s Warning' % progname)
                box.setText('<b>Overwrite Warning</b>')
                box.setIcon(QtGui.QMessageBox.Warning)
                box.setFont(font_standard)
                box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
                box.setInformativeText(msg)
                reply = box.exec_()
                if reply == QtGui.QMessageBox.Yes:
                    P4.btn_finOut.setText('Compile Results')
                    P4.recompute = 0
                    P5.progress5.setValue(0)
                    P5.progress6.setValue(0)
                    Erg1dMain.progress7.setValue(0)
                    P5.button_CC.setText('Find Conformational Coordinates')
                    P5.button_erg.setText('Energy Landscape')
                    Erg1dMain.button_traj.setText('Compute 3D Trajectories')
                    Erg1dMain.reprepare = 0
                    P5.entry_opt.setDisabled(False)
                    P5.entry_proc.setDisabled(False)
                    P5.entry_temp.setDisabled(False)
                    P5.button_CC.setDisabled(False)
                    P5.button_erg.setDisabled(True)
                    Erg1dMain.button_traj.setDisabled(True)
                    P5.button_toP6.setDisabled(True)
                    tabs.setTabEnabled(4, False)
                    tabs.setTabEnabled(5, False)
                    p.resProj = 5 #revert progress prior to ever moving to Compilation tab
                    set_params.op(0) #send new GUI data to user parameters file

                    # =============================================
                    # Hard-remove pre-existing folders:
                    shutil.rmtree(p.CC_meas_dir)
                    shutil.rmtree(p.CC_OF_dir)
                    shutil.rmtree(p.EL_dir)
                    prepOutDir = os.path.join(p.out_dir,'bin')
                    shutil.rmtree(prepOutDir)
                    shutil.rmtree(p.traj_dir)

                    time.sleep(1)
                    os.makedirs(p.CC_meas_dir)
                    os.makedirs(p.CC_OF_dir)
                    os.makedirs(p.EL_dir)
                    os.makedirs(p.OM_dir)
                    os.makedirs(prepOutDir)
                    os.makedirs(p.traj_dir)
                    os.makedirs(p.CC_meas_prog) #progress bar folder
                    os.makedirs(p.EL_prog) #progress bar folder
                    # =========================================================
                    # repeat of code from below (if P4.recompute == 0):
                    P3.user_dimensions = P3.entry_dim.value()
                    p.dim = P3.user_dimensions
                    set_params.op(0) #send new GUI data to user parameters file

                    if P3.dictGen == True: #if P4 widget-dictionary has been created
                        if P3.user_dimensions == 1:
                            # enable correct subtab for tab5:
                            erg_tabs.setTabEnabled(1, False)
                            erg_tabs.setTabEnabled(0, True)
                            erg_tabs.setCurrentIndex(0)
                            Erg1dMain.chooseCC.setDisabled(True)
                            Erg1dMain.chooseCC.setCurrentIndex(0)
                            # remove all anchor widgets:
                            P4.layoutB.removeWidget(P4.reactCoord1All[P4.PrD_hist])
                            P4.layoutB.removeWidget(P4.senses1All[P4.PrD_hist])
                            P4.layoutB.removeWidget(P4.reactCoord2All[P4.PrD_hist])
                            P4.layoutB.removeWidget(P4.senses2All[P4.PrD_hist])
                            P4.layoutB.removeWidget(P4.anchorsAll[P4.PrD_hist])
                            P4.reactCoord1All[P4.PrD_hist].close()
                            P4.senses1All[P4.PrD_hist].close()
                            P4.reactCoord2All[P4.PrD_hist].close()
                            P4.senses2All[P4.PrD_hist].close()
                            P4.anchorsAll[P4.PrD_hist].close()
                            # add back and rearrange 1D widgets:
                            P4.layoutB.addWidget(P4.reactCoord1All[P4.user_PrD], 8, 2, 1, 1)
                            P4.layoutB.addWidget(P4.senses1All[P4.user_PrD], 8, 3, 1, 1)
                            P4.layoutB.addWidget(P4.anchorsAll[P4.user_PrD], 8, 4, 1, 1)
                            P4.reactCoord1All[P4.user_PrD].show()
                            P4.senses1All[P4.user_PrD].show()
                            P4.anchorsAll[P4.user_PrD].show()

                            for i in range(1,P3.PrD_total+1):
                                P4.reactCoord2All[i].setDisabled(True)
                                P4.senses2All[i].setToolTip('')
                                P4.senses2All[i].setDisabled(True)

                        elif P3.user_dimensions == 2:
                            # enable correct subtab for tab5:
                            erg_tabs.setTabEnabled(0, True)
                            erg_tabs.setTabEnabled(1, True)
                            erg_tabs.setCurrentIndex(1)
                            Erg1dMain.chooseCC.setDisabled(False)
                            Erg1dMain.chooseCC.setCurrentIndex(0)

                            # remove all anchor widgets:
                            P4.layoutB.removeWidget(P4.reactCoord1All[P4.PrD_hist])
                            P4.layoutB.removeWidget(P4.senses1All[P4.PrD_hist])
                            P4.layoutB.removeWidget(P4.reactCoord2All[P4.PrD_hist])
                            P4.layoutB.removeWidget(P4.senses2All[P4.PrD_hist])
                            P4.layoutB.removeWidget(P4.anchorsAll[P4.PrD_hist])
                            P4.reactCoord1All[P4.PrD_hist].close()
                            P4.senses1All[P4.PrD_hist].close()
                            P4.reactCoord2All[P4.PrD_hist].close()
                            P4.senses2All[P4.PrD_hist].close()
                            P4.anchorsAll[P4.PrD_hist].close()
                            # add back and rearrange 2D widgets:
                            P4.layoutB.addWidget(P4.reactCoord1All[P4.user_PrD], 8, 1, 1, 1)
                            P4.layoutB.addWidget(P4.senses1All[P4.user_PrD], 8, 2, 1, 1)
                            P4.layoutB.addWidget(P4.reactCoord2All[P4.user_PrD], 8, 3, 1, 1)
                            P4.layoutB.addWidget(P4.senses2All[P4.user_PrD], 8, 4, 1, 1)
                            P4.layoutB.addWidget(P4.anchorsAll[P4.user_PrD], 8, 5, 1, 1)
                            P4.reactCoord1All[P4.user_PrD].show()
                            P4.senses1All[P4.user_PrD].show()
                            P4.reactCoord2All[P4.user_PrD].show()
                            P4.senses2All[P4.user_PrD].show()
                            P4.anchorsAll[P4.user_PrD].show()

                            for i in range(1,P3.PrD_total+1):
                                if P4.anchorsAll[i].isChecked():
                                    P4.reactCoord2All[i].setDisabled(True)
                                    P4.senses2All[i].setDisabled(True)
                                else:
                                    P4.reactCoord2All[i].setDisabled(False)
                                    P4.senses2All[i].setDisabled(False)
                                P4.senses2All[i].setToolTip('CC2: Confirm sense for selected topos.')
                                P4.reactCoord2All[i].setPrefix('CC2: %s' % (u"\u03A8"))
                                P4.reactCoord2All[i].setMinimum(1)
                                P4.reactCoord2All[i].setMaximum(p.num_psis)
                                if P4.reactCoord1All[i].value() == P4.reactCoord2All[i].value():
                                    if P4.reactCoord1All[i].value() > 1:
                                        P4.reactCoord2All[i].setValue(1)
                                    elif P4.reactCoord1All[i].value() == 1:
                                        P4.reactCoord2All[i].setValue(2)
                else:
                    P3.entry_dim.blockSignals(True)
                    P3.entry_dim.setValue(dim_temp)
                    P3.entry_dim.blockSignals(False)
                    pass
            else:
                P3.user_dimensions = P3.entry_dim.value()
                p.dim = P3.user_dimensions
                set_params.op(0) #send new GUI data to user parameters file

                if P3.dictGen == True: #if P4 widget-dictionary has been created
                    if P3.user_dimensions == 1:
                        # enable correct subtab for tab5:
                        erg_tabs.setTabEnabled(1, False)
                        erg_tabs.setTabEnabled(0, True)
                        erg_tabs.setCurrentIndex(0)
                        Erg1dMain.chooseCC.setDisabled(True)
                        Erg1dMain.chooseCC.setCurrentIndex(0)

                        # remove all anchor widgets:
                        P4.layoutB.removeWidget(P4.reactCoord1All[P4.PrD_hist])
                        P4.layoutB.removeWidget(P4.senses1All[P4.PrD_hist])
                        P4.layoutB.removeWidget(P4.reactCoord2All[P4.PrD_hist])
                        P4.layoutB.removeWidget(P4.senses2All[P4.PrD_hist])
                        P4.layoutB.removeWidget(P4.anchorsAll[P4.PrD_hist])
                        P4.reactCoord1All[P4.PrD_hist].close()
                        P4.senses1All[P4.PrD_hist].close()
                        P4.reactCoord2All[P4.PrD_hist].close()
                        P4.senses2All[P4.PrD_hist].close()
                        P4.anchorsAll[P4.PrD_hist].close()
                        # add back and rearrange 1D widgets:
                        P4.layoutB.addWidget(P4.reactCoord1All[P4.user_PrD], 8, 2, 1, 1)
                        P4.layoutB.addWidget(P4.senses1All[P4.user_PrD], 8, 3, 1, 1)
                        P4.layoutB.addWidget(P4.anchorsAll[P4.user_PrD], 8, 4, 1, 1)
                        P4.reactCoord1All[P4.user_PrD].show()
                        P4.senses1All[P4.user_PrD].show()
                        P4.anchorsAll[P4.user_PrD].show()

                        for i in range(1,P3.PrD_total+1):
                            P4.reactCoord2All[i].setDisabled(True)
                            P4.senses2All[i].setToolTip('')
                            P4.senses2All[i].setDisabled(True)

                    elif P3.user_dimensions == 2:
                        # enable correct subtab for tab5:
                        erg_tabs.setTabEnabled(0, True)
                        erg_tabs.setTabEnabled(1, True)
                        erg_tabs.setCurrentIndex(1)
                        Erg1dMain.chooseCC.setDisabled(False)
                        Erg1dMain.chooseCC.setCurrentIndex(0)
                        # remove all anchor widgets:
                        P4.layoutB.removeWidget(P4.reactCoord1All[P4.PrD_hist])
                        P4.layoutB.removeWidget(P4.senses1All[P4.PrD_hist])
                        P4.layoutB.removeWidget(P4.reactCoord2All[P4.PrD_hist])
                        P4.layoutB.removeWidget(P4.senses2All[P4.PrD_hist])
                        P4.layoutB.removeWidget(P4.anchorsAll[P4.PrD_hist])
                        P4.reactCoord1All[P4.PrD_hist].close()
                        P4.senses1All[P4.PrD_hist].close()
                        P4.reactCoord2All[P4.PrD_hist].close()
                        P4.senses2All[P4.PrD_hist].close()
                        P4.anchorsAll[P4.PrD_hist].close()
                        # add back and rearrange 2D widgets:
                        P4.layoutB.addWidget(P4.reactCoord1All[P4.user_PrD], 8, 1, 1, 1)
                        P4.layoutB.addWidget(P4.senses1All[P4.user_PrD], 8, 2, 1, 1)
                        P4.layoutB.addWidget(P4.reactCoord2All[P4.user_PrD], 8, 3, 1, 1)
                        P4.layoutB.addWidget(P4.senses2All[P4.user_PrD], 8, 4, 1, 1)
                        P4.layoutB.addWidget(P4.anchorsAll[P4.user_PrD], 8, 5, 1, 1)
                        P4.reactCoord1All[P4.user_PrD].show()
                        P4.senses1All[P4.user_PrD].show()
                        P4.reactCoord2All[P4.user_PrD].show()
                        P4.senses2All[P4.user_PrD].show()
                        P4.anchorsAll[P4.user_PrD].show()

                        for i in range(1,P3.PrD_total+1):
                            if P4.anchorsAll[i].isChecked():
                                P4.reactCoord2All[i].setDisabled(True)
                                P4.senses2All[i].setDisabled(True)
                            else:
                                P4.reactCoord2All[i].setDisabled(False)
                                P4.senses2All[i].setDisabled(False)
                            P4.senses2All[i].setToolTip('CC2: Confirm sense for selected topos.')
                            P4.reactCoord2All[i].setPrefix('CC2: %s' % (u"\u03A8"))
                            P4.reactCoord2All[i].setMinimum(1)
                            P4.reactCoord2All[i].setMaximum(p.num_psis)
                            if P4.reactCoord1All[i].value() == P4.reactCoord2All[i].value():
                                if P4.reactCoord1All[i].value() > 1:
                                    P4.reactCoord2All[i].setValue(1)
                                elif P4.reactCoord1All[i].value() == 1:
                                    P4.reactCoord2All[i].setValue(2)
        
        # forced space top:
        self.label_spaceT = QtGui.QLabel("")
        self.label_spaceT.setFont(font_standard)
        self.label_spaceT.setMargin(0)
        layout.addWidget(self.label_spaceT, 0, 0, 1, 7, QtCore.Qt.AlignVCenter)
        self.label_spaceT.show()

        # main outline:
        self.label_edgeMain = QtGui.QLabel('')
        self.label_edgeMain.setMargin(20)
        self.label_edgeMain.setLineWidth(1)
        self.label_edgeMain.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edgeMain, 0, 0, 13, 8)
        self.label_edgeMain.show()

        self.label_edge0 = QtGui.QLabel('')
        self.label_edge0.setMargin(20)
        self.label_edge0.setLineWidth(1)
        self.label_edge0.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge0, 1, 1, 1, 2)
        self.label_edge0.show()

        self.label_edge0a = QtGui.QLabel('')
        self.label_edge0a.setMargin(20)
        self.label_edge0a.setLineWidth(1)
        self.label_edge0a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge0a, 1, 1, 1, 1)
        self.label_edge0a.show()

        self.label_proc = QtGui.QLabel('Processors')
        self.label_proc.setFont(font_standard)
        self.label_proc.setMargin(20)
        self.label_proc.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        self.label_proc.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_proc, 1, 1, 1, 1)
        self.label_proc.show()

        P3.entry_proc = QtGui.QSpinBox(self)
        P3.entry_proc.setMinimum(1)
        if p.machinefile:
            P3.entry_proc.setMaximum(1000)
        else:
            P3.entry_proc.setMaximum(multiprocessing.cpu_count())
        P3.entry_proc.valueChanged.connect(choose_processors)
        P3.entry_proc.setStyleSheet("QSpinBox { width : 100px }")
        P3.entry_proc.setToolTip('The number of processors to use in parallel.')
        layout.addWidget(P3.entry_proc, 1, 2, 1, 1, QtCore.Qt.AlignLeft)
        P3.entry_proc.show()

        self.label_edge00 = QtGui.QLabel('')
        self.label_edge00.setMargin(20)
        self.label_edge00.setLineWidth(1)
        self.label_edge00.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge00, 1, 3, 1, 2)
        self.label_edge00.show()

        self.label_edge00a = QtGui.QLabel('')
        self.label_edge00a.setMargin(20)
        self.label_edge00a.setLineWidth(1)
        self.label_edge00a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge00a, 1, 3, 1, 1)
        self.label_edge00a.show()

        self.label_psi = QtGui.QLabel('Eigenvectors')
        self.label_psi.setFont(font_standard)
        self.label_psi.setMargin(20)
        self.label_psi.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        self.label_psi.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_psi, 1, 3, 1, 1)
        self.label_psi.show()

        P3.entry_psi = QtGui.QSpinBox(self)
        P3.entry_psi.setMinimum(1)
        P3.entry_psi.setMaximum(8)
        P3.entry_psi.setValue(8)
        P3.entry_psi.valueChanged.connect(choose_psi)
        P3.entry_psi.setStyleSheet("QSpinBox { width : 100px }")
        P3.entry_psi.setToolTip('The number of DM eigenvectors to consider for NLSA.')
        layout.addWidget(P3.entry_psi, 1, 4, 1, 1, QtCore.Qt.AlignLeft)
        P3.entry_psi.show()

        self.label_edge000 = QtGui.QLabel('')
        self.label_edge000.setMargin(20)
        self.label_edge000.setLineWidth(1)
        self.label_edge000.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge000, 1, 5, 1, 2)
        self.label_edge000.show()
        
        self.label_edge000a = QtGui.QLabel('')
        self.label_edge000a.setMargin(20)
        self.label_edge000a.setLineWidth(1)
        self.label_edge000a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge000a, 1, 5, 1, 1)
        self.label_edge000a.show()
        
        self.label_dim = QtGui.QLabel('Dimensions')
        self.label_dim.setFont(font_standard)
        self.label_dim.setMargin(20)
        self.label_dim.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        self.label_dim.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_dim, 1, 5, 1, 1)
        self.label_dim.show()

        P3.entry_dim = QtGui.QSpinBox(self)
        P3.entry_dim.setMinimum(1)
        if Beta is True:
            P3.entry_dim.setMaximum(1)
            P3.entry_dim.setValue(1)
            P3.entry_dim.setDisabled(True)
        else:
            P3.entry_dim.setMaximum(2)
            P3.entry_dim.setValue(2)
            P3.entry_dim.setDisabled(False)
        P3.entry_dim.valueChanged.connect(choose_dimensions)
        P3.entry_dim.setToolTip('The number of orthogonal conformational coordinates to compare within the energy landscape.')
        P3.entry_dim.setStyleSheet("QSpinBox { width : 100px }")
        layout.addWidget(P3.entry_dim, 1, 6, 1, 1, QtCore.Qt.AlignLeft)
        P3.entry_dim.show()
            
        # distances progress:
        P3.button_dist = QtGui.QPushButton('Distance Calculation', self)
        P3.button_dist.clicked.connect(self.start_task1)
        layout.addWidget(P3.button_dist, 3, 1, 1, 2)
        P3.button_dist.setDisabled(False)
        P3.button_dist.show()

        P3.progress1 = QtGui.QProgressBar(minimum=0, maximum=100, value=0)
        self.progress1Changed.connect(self.on_progress1Changed)       
        layout.addWidget(P3.progress1, 3, 3, 1, 4)
        P3.progress1.show()

        # eigenvectors progress:
        self.label_Hline1 = QtGui.QLabel("")
        self.label_Hline1.setFont(font_standard)
        self.label_Hline1.setMargin(0)
        self.label_Hline1.setFrameStyle(QtGui.QFrame.HLine | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_Hline1, 4, 1, 1, 6, QtCore.Qt.AlignVCenter)
        self.label_Hline1.show()

        P3.button_eig = QtGui.QPushButton('Embedding', self)
        P3.button_eig.clicked.connect(self.start_task2)
        layout.addWidget(P3.button_eig, 5, 1, 1, 2)
        P3.button_eig.setDisabled(True)
        P3.button_eig.show()

        P3.progress2 = QtGui.QProgressBar(minimum=0, maximum=100, value=0)
        self.progress2Changed.connect(self.on_progress2Changed)       
        layout.addWidget(P3.progress2, 5, 3, 1, 4)
        P3.progress2.show()

        # spectral anaylsis progress:
        self.label_Hline2 = QtGui.QLabel("")
        self.label_Hline2.setFont(font_standard)
        self.label_Hline2.setMargin(0)
        self.label_Hline2.setFrameStyle(QtGui.QFrame.HLine | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_Hline2, 6, 1, 1, 6, QtCore.Qt.AlignVCenter)
        self.label_Hline2.show()
        
        P3.button_psi = QtGui.QPushButton('Spectral Analysis', self)
        P3.button_psi.clicked.connect(self.start_task3)
        layout.addWidget(P3.button_psi, 7, 1, 1, 2)
        P3.button_psi.setDisabled(True)
        P3.button_psi.show()

        P3.progress3 = QtGui.QProgressBar(minimum=0, maximum=100, value=0)
        self.progress3Changed.connect(self.on_progress3Changed)       
        layout.addWidget(P3.progress3, 7, 3, 1, 4)
        P3.progress3.show()

        # nlsa movie progress: 
        self.label_Hline3 = QtGui.QLabel("")
        self.label_Hline3.setFont(font_standard)
        self.label_Hline3.setMargin(0)
        self.label_Hline3.setFrameStyle(QtGui.QFrame.HLine | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_Hline3, 8, 1, 1, 6, QtCore.Qt.AlignVCenter)
        self.label_Hline3.show()
        
        P3.button_nlsa = QtGui.QPushButton('Compile 2D Movies', self)
        P3.button_nlsa.clicked.connect(self.start_task4)
        layout.addWidget(P3.button_nlsa, 9, 1, 1, 2)
        P3.button_nlsa.setDisabled(True)
        P3.button_nlsa.show()

        P3.progress4 = QtGui.QProgressBar(minimum=0, maximum=100, value=0)
        self.progress4Changed.connect(self.on_progress4Changed)       
        layout.addWidget(P3.progress4, 9, 3, 1, 4)
        P3.progress4.show()

        self.label_Hline3 = QtGui.QLabel("")
        self.label_Hline3.setFont(font_standard)
        self.label_Hline3.setMargin(0)
        self.label_Hline3.setFrameStyle(QtGui.QFrame.HLine | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_Hline3, 10, 1, 1, 6, QtCore.Qt.AlignVCenter)
        self.label_Hline3.show()

        P3.button_toP4 = QtGui.QPushButton('View Eigenvectors', self)
        P3.button_toP4.clicked.connect(self.calcP4)
        layout.addWidget(P3.button_toP4, 11, 3, 1, 2)
        P3.button_toP4.setDisabled(True)
        P3.button_toP4.show()

        # extend spacing:
        self.label_space = QtGui.QLabel("")
        self.label_space.setFont(font_standard)
        self.label_space.setMargin(0)
        layout.addWidget(self.label_space, 11, 0, 5, 4, QtCore.Qt.AlignVCenter)
        self.label_space.show()

    # =============================================================================
    # calcP4() functions:
    # =============================================================================
    def unique_eigs(self):
        if P3.user_dimensions == 2:
            if P4.reactCoord1All[P4.user_PrD].value() == P4.reactCoord2All[P4.user_PrD].value():
                P4.anchorsAll[P4.user_PrD].setDisabled(True)
            else:
                P4.anchorsAll[P4.user_PrD].setDisabled(False)
        elif P3.user_dimensions == 1:
            P4.anchorsAll[P4.user_PrD].setDisabled(False)

    def set_anchor(self):
        if P4.anchorsAll[P4.user_PrD].isChecked():
            P4.reactCoord1All[P4.user_PrD].setDisabled(True)
            P4.senses1All[P4.user_PrD].setDisabled(True)
            if P3.user_dimensions == 2:
                P4.reactCoord2All[P4.user_PrD].setDisabled(True)
                P4.senses2All[P4.user_PrD].setDisabled(True)

            # update S2 anchors plot:
            P1.x3.append(P1.x2[P4.user_PrD-1])
            P1.y3.append(P1.y2[P4.user_PrD-1])
            P1.z3.append(P1.z2[P4.user_PrD-1])
            P1.a3.append(P4.user_PrD)

            zoom = P4.viz2.view_anglesP4(dialog = False)
            P4.viz2.update_scene3()
            P4.viz2.update_viewP4(azimuth = float(P3.phi[(P4.user_PrD)-1]),
                                elevation = float(P3.theta[(P4.user_PrD)-1]),
                                distance = zoom)

        else: #check=off
            P4.reactCoord1All[P4.user_PrD].setDisabled(False)
            P4.senses1All[P4.user_PrD].setDisabled(False)
            if P3.user_dimensions == 2:
                P4.reactCoord2All[P4.user_PrD].setDisabled(False)
                P4.senses2All[P4.user_PrD].setDisabled(False)
            # update S2 anchors plot:
            idx = 0
            for i in P1.a3:
                if int(i) == int(P4.user_PrD):
                    P1.x3.pop(idx)
                    P1.y3.pop(idx)
                    P1.z3.pop(idx)
                    P1.a3.pop(idx)
                idx += 1

            zoom = P4.viz2.view_anglesP4(dialog = False)
            P4.viz2.update_scene3()
            P4.viz2.update_viewP4(azimuth = float(P3.phi[(P4.user_PrD)-1]),
                                elevation = float(P3.theta[(P4.user_PrD)-1]),
                                distance = zoom)

    def set_trash(self):
        if P4.trashAll[P4.user_PrD].isChecked():
            P4.anchorsAll[P4.user_PrD].setChecked(False)
            P4.anchorsAll[P4.user_PrD].setDisabled(True)

            P4.reactCoord1All[P4.user_PrD].setDisabled(True)
            P4.senses1All[P4.user_PrD].setDisabled(True)
            if P3.user_dimensions == 2:
                P4.reactCoord2All[P4.user_PrD].setDisabled(True)
                P4.senses2All[P4.user_PrD].setDisabled(True)
            # update S2 trash plot:
            P1.x4.append(P1.x2[P4.user_PrD-1])
            P1.y4.append(P1.y2[P4.user_PrD-1])
            P1.z4.append(P1.z2[P4.user_PrD-1])
            P1.a4.append(P4.user_PrD)
            # update S2 anchors plot:
            idx = 0
            for i in P1.a3:
                if int(i) == int(P4.user_PrD):
                    P1.x3.pop(idx)
                    P1.y3.pop(idx)
                    P1.z3.pop(idx)
                    P1.a3.pop(idx)
                idx += 1

            zoom = P4.viz2.view_anglesP4(dialog = False)
            P4.viz2.update_scene3()
            P4.viz2.update_viewP4(azimuth = float(P3.phi[(P4.user_PrD)-1]),
                                elevation = float(P3.theta[(P4.user_PrD)-1]),
                                distance = zoom)

        else:
            P4.anchorsAll[P4.user_PrD].setDisabled(False)
            P4.reactCoord1All[P4.user_PrD].setDisabled(False)
            P4.senses1All[P4.user_PrD].setDisabled(False)
            if P3.user_dimensions == 2:
                P4.reactCoord2All[P4.user_PrD].setDisabled(False)
                P4.senses2All[P4.user_PrD].setDisabled(False)
            # update S2 trash plot:
            idx = 0
            for i in P1.a4:
                if int(i) == int(P4.user_PrD):
                    P1.x4.pop(idx)
                    P1.y4.pop(idx)
                    P1.z4.pop(idx)
                    P1.a4.pop(idx)
                idx += 1

            zoom = P4.viz2.view_anglesP4(dialog = False)
            P4.viz2.update_scene3()
            P4.viz2.update_viewP4(azimuth = float(P3.phi[(P4.user_PrD)-1]),
                                elevation = float(P3.theta[(P4.user_PrD)-1]),
                                distance = zoom)
    # end of calcP4 functions
    # =========================================================================

    def calcP4(self):
        # =====================================================================
        # PREPARE P4:
        # =====================================================================
        # PD threshold on Alignment data:
        fname = os.path.join(P1.user_directory, 'outputs_{}/topos/Euler_PrD/PrD_map.txt'.format(p.proj_name))
        data = []
        with open(fname) as values:
            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                data.append(column)

        P3.prd = data[0]
        P3.theta = data[1]
        P3.phi = data[2]
        psi = data[3]
        x = data[4]
        y = data[5]
        z = data[6]
        c = data[7] #colors of connected components

        xyzc = np.column_stack((x,y,z,c))

        P1.x2 = []
        P1.y2 = []
        P1.z2 = []
        P3.col = []

        for i,j,k,l in xyzc:
            P1.x2.append(float(i))
            P1.y2.append(float(j))
            P1.z2.append(float(k))
            P3.col.append(int(l))

        P3.PrD_total = len(P1.x2)
        P4.entry_PrD.setMaximum(P3.PrD_total) #updates P4
        P4.entry_PrD.setSuffix(' / %s' % (P3.PrD_total))
        P4.viz2.update_scene3() #ZULU redundant when resuming?
        P4.viz2.update_viewP4(float(P3.phi[0]),
                            float(P3.theta[0]),
                            P4.viz2.view_anglesP4(dialog = False)) #grab current zoom
        EigValCanvas().EigValRead() #read in eigenvalue spectrum for current PrD
        # =============================================================================
        # Create dictionary for anchor widgets:
        # =============================================================================
        P3.dictGen = True
        P4.origEmbedFile = os.path.join(P1.user_directory, 'outputs_{}/topos/Euler_PrD/PrD_embeds.txt'.format(p.proj_name))
        for i in range(1,P3.PrD_total+1):
            CC1 = QtGui.QSpinBox(self)
            CC1.setMinimum(1)
            CC1.setMaximum(p.num_psis)
            CC1.setFont(font_standard)
            CC1.setPrefix('CC1: %s' % (u"\u03A8"))
            CC1.valueChanged.connect(self.unique_eigs)
            if i > 1:
                CC1.hide()
            P4.reactCoord1All[i] = CC1

            CC2 = QtGui.QSpinBox(self)
            CC2.setMinimum(1)
            CC2.setMaximum(p.num_psis)
            CC2.setFont(font_standard)
            CC2.setPrefix('CC2: %s' % (u"\u03A8"))
            CC2.valueChanged.connect(self.unique_eigs)
            if P3.user_dimensions == 1:
                CC2.hide()
            else:
                if i > 1:
                    CC2.hide()
            P4.reactCoord2All[i] = CC2

            sense1 = QtGui.QComboBox(self)
            sense1.addItem('S1: FWD')
            sense1.addItem('S1: REV')
            sense1.setFont(font_standard)
            sense1.setToolTip('CC1: Confirm sense for selected topos.')
            if i > 1:
                sense1.hide()
            P4.senses1All[i] = sense1
            P4.senses1All[i].setToolTip('CC1: Confirm sense for selected topos.')

            sense2 = QtGui.QComboBox(self)
            sense2.addItem('S2: FWD')
            sense2.addItem('S2: REV')
            sense2.setFont(font_standard)
            sense2.setDisabled(True)
            sense2.setToolTip('')
            if P3.user_dimensions == 1:
                sense2.hide()
            else:
                if i > 1:
                    sense2.hide()
            P4.senses2All[i] = sense2
            P4.senses2All[i].setToolTip('CC2: Confirm sense for selected topos.')

            anchor = QtGui.QCheckBox('Set Anchor', self)
            anchor.setChecked(False)
            anchor.setFont(font_standard)
            anchor.setToolTip('Check to make the current PD an anchor node.')
            anchor.stateChanged.connect(self.set_anchor)
            if i > 1:
                anchor.hide()
            P4.anchorsAll[i] = anchor

            trash = QtGui.QCheckBox('Remove PD', self)
            trash.setChecked(False)
            trash.setFont(font_standard)
            trash.setToolTip('Check to remove the current PD from the final reconstruction.')
            trash.stateChanged.connect(self.set_trash)
            if i > 1:
                trash.hide()
            P4.trashAll[i] = trash
            # keep track of manifold re-embeddings:
            if os.path.isfile(P4.origEmbedFile) is False:
                P4.origEmbed.append(int(1)) #int(1) is True: PD has its original embedding

        if os.path.isfile(P4.origEmbedFile) is False:
            np.savetxt(P4.origEmbedFile, P4.origEmbed, fmt='%i')
        # =====================================================================
        # Update trash section widgets:
        # =====================================================================
        P4.layoutL.removeWidget(P4.trashAll[P4.PrD_hist])
        P4.trashAll[P4.PrD_hist].close()
        P4.layoutL.addWidget(P4.trashAll[P4.user_PrD], 6, 5, 1, 2, QtCore.Qt.AlignCenter)
        P4.trashAll[P4.user_PrD].show()
        # =============================================================================
        # Update anchor section widgets based on dimensions (set prior):
        # =============================================================================
        if P3.user_dimensions == 1:
            P4.layoutB.removeWidget(P4.reactCoord1All[P4.PrD_hist])
            P4.reactCoord1All[P4.PrD_hist].close()
            P4.layoutB.addWidget(P4.reactCoord1All[P4.user_PrD], 8, 2, 1, 1)
            P4.reactCoord1All[P4.user_PrD].show()

            P4.layoutB.removeWidget(P4.senses1All[P4.PrD_hist])
            P4.senses1All[P4.PrD_hist].close()
            P4.layoutB.addWidget(P4.senses1All[P4.user_PrD], 8, 3, 1, 1)
            P4.senses1All[P4.user_PrD].show()

            P4.layoutB.removeWidget(P4.anchorsAll[P4.PrD_hist])
            P4.anchorsAll[P4.PrD_hist].close()
            P4.layoutB.addWidget(P4.anchorsAll[P4.user_PrD], 8, 4, 1, 1)
            P4.anchorsAll[P4.user_PrD].show()

        if P3.user_dimensions == 2:
            P4.layoutB.removeWidget(P4.reactCoord1All[P4.PrD_hist])
            P4.reactCoord1All[P4.PrD_hist].close()
            P4.layoutB.addWidget(P4.reactCoord1All[P4.user_PrD], 8, 1, 1, 1)
            P4.reactCoord1All[P4.user_PrD].show()

            P4.layoutB.removeWidget(P4.senses1All[P4.PrD_hist])
            P4.senses1All[P4.PrD_hist].close()
            P4.layoutB.addWidget(P4.senses1All[P4.user_PrD], 8, 2, 1, 1)
            P4.senses1All[P4.user_PrD].show()

            P4.layoutB.removeWidget(P4.reactCoord2All[P4.PrD_hist])
            P4.reactCoord2All[P4.PrD_hist].close()
            P4.layoutB.addWidget(P4.reactCoord2All[P4.user_PrD], 8, 3, 1, 1)
            P4.reactCoord2All[P4.user_PrD].show()

            P4.layoutB.removeWidget(P4.senses2All[P4.PrD_hist])
            P4.senses2All[P4.PrD_hist].close()
            P4.layoutB.addWidget(P4.senses2All[P4.user_PrD], 8, 4, 1, 1)
            P4.senses2All[P4.user_PrD].show()

            P4.layoutB.removeWidget(P4.anchorsAll[P4.PrD_hist])
            P4.anchorsAll[P4.PrD_hist].close()
            P4.layoutB.addWidget(P4.anchorsAll[P4.user_PrD], 8, 5, 1, 1)
            P4.anchorsAll[P4.user_PrD].show()

            for i in range(1,P3.PrD_total+1): #only needed for dim > 1
                if P4.anchorsAll[i].isChecked():
                    P4.reactCoord1All[i].setDisabled(True)
                    P4.reactCoord2All[i].setDisabled(True)
                    P4.senses1All[i].setDisabled(True)
                    P4.senses2All[i].setDisabled(True)
                else:
                    P4.reactCoord1All[i].setDisabled(False)
                    P4.reactCoord2All[i].setDisabled(False)
                    P4.senses1All[i].setDisabled(False)
                    P4.senses2All[i].setDisabled(False)
                
                    if P4.reactCoord1All[i].value() == P4.reactCoord2All[i].value():
                        if P4.reactCoord1All[i].value() > 1:
                            P4.reactCoord2All[i].setValue(1)
                        elif P4.reactCoord1All[i].value() == 1:
                            P4.reactCoord2All[i].setValue(2) #ZULU: why doesn't this work?
        # =====================================================================                 
        # update topos cover images:
        # =====================================================================
        # create blank image if topos file doesn't exists
        # =====================================================================
        picDir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_1.png' % (p.proj_name,P4.user_PrD))
        picImg = Image.open(picDir)
        picSize = picImg.size
        blank = np.zeros([picSize[0],picSize[1],3], dtype=np.uint8)
        blank.fill(0)
        blank = QtGui.QImage(blank, blank.shape[1],\
        blank.shape[0], blank.shape[1] * 3,QtGui.QImage.Format_RGB888)
        blankpix = QtGui.QPixmap(blank)
        # =====================================================================
        P4.user_PrD = P4.entry_PrD.value()

        P4.pic1Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_1.png' % (p.proj_name,P4.user_PrD))
        if os.path.isfile(P4.pic1Dir):
            P4.label_pic1.setPixmap(QtGui.QPixmap(P4.pic1Dir))
            P4.button_pic1.setDisabled(False)
            P4.reactCoord1All[P4.user_PrD].setMaximum(1)
            P4.reactCoord2All[P4.user_PrD].setMaximum(1)
        else:
            P4.label_pic1.setPixmap(QtGui.QPixmap(blankpix))
            P4.button_pic1.setDisabled(True)

        P4.pic2Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_2.png' % (p.proj_name,P4.user_PrD))
        if os.path.isfile(P4.pic2Dir):
            P4.label_pic2.setPixmap(QtGui.QPixmap(P4.pic2Dir))
            P4.button_pic2.setDisabled(False)
            P4.reactCoord1All[P4.user_PrD].setMaximum(2)
            P4.reactCoord2All[P4.user_PrD].setMaximum(2)
        else:
            P4.label_pic2.setPixmap(QtGui.QPixmap(blankpix))
            P4.button_pic2.setDisabled(True)

        P4.pic3Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_3.png' % (p.proj_name,P4.user_PrD))
        if os.path.isfile(P4.pic3Dir):
            P4.label_pic3.setPixmap(QtGui.QPixmap(P4.pic3Dir))
            P4.button_pic3.setDisabled(False)
            P4.reactCoord1All[P4.user_PrD].setMaximum(3)
            P4.reactCoord2All[P4.user_PrD].setMaximum(3)
        else:
            P4.label_pic3.setPixmap(QtGui.QPixmap(blankpix))
            P4.button_pic3.setDisabled(True)

        P4.pic4Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_4.png' % (p.proj_name,P4.user_PrD))
        if os.path.isfile(P4.pic4Dir):
            P4.label_pic4.setPixmap(QtGui.QPixmap(P4.pic4Dir))
            P4.button_pic4.setDisabled(False)
            P4.reactCoord1All[P4.user_PrD].setMaximum(4)
            P4.reactCoord2All[P4.user_PrD].setMaximum(4)
        else:
            P4.label_pic4.setPixmap(QtGui.QPixmap(blankpix))
            P4.button_pic4.setDisabled(True)

        P4.pic5Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_5.png' % (p.proj_name,P4.user_PrD))
        if os.path.isfile(P4.pic5Dir):
            P4.label_pic5.setPixmap(QtGui.QPixmap(P4.pic5Dir))
            P4.button_pic5.setDisabled(False)
            P4.reactCoord1All[P4.user_PrD].setMaximum(5)
            P4.reactCoord2All[P4.user_PrD].setMaximum(5)
        else:
            P4.label_pic5.setPixmap(QtGui.QPixmap(blankpix))
            P4.button_pic5.setDisabled(True)

        P4.pic6Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_6.png' % (p.proj_name,P4.user_PrD))
        if os.path.isfile(P4.pic6Dir):
            P4.label_pic6.setPixmap(QtGui.QPixmap(P4.pic6Dir))
            P4.button_pic6.setDisabled(False)
            P4.reactCoord1All[P4.user_PrD].setMaximum(6)
            P4.reactCoord2All[P4.user_PrD].setMaximum(6)
        else:
            P4.label_pic6.setPixmap(QtGui.QPixmap(blankpix))
            P4.button_pic6.setDisabled(True)

        P4.pic7Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_7.png' % (p.proj_name,P4.user_PrD))
        if os.path.isfile(P4.pic7Dir):
            P4.label_pic7.setPixmap(QtGui.QPixmap(P4.pic7Dir))
            P4.button_pic7.setDisabled(False)
            P4.reactCoord1All[P4.user_PrD].setMaximum(7)
            P4.reactCoord2All[P4.user_PrD].setMaximum(7)
        else:
            P4.label_pic7.setPixmap(QtGui.QPixmap(blankpix))
            P4.button_pic7.setDisabled(True)

        P4.pic8Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_8.png' % (p.proj_name,P4.user_PrD))
        if os.path.isfile(P4.pic8Dir):
            P4.label_pic8.setPixmap(QtGui.QPixmap(P4.pic8Dir))
            P4.button_pic8.setDisabled(False)
            P4.reactCoord1All[P4.user_PrD].setMaximum(p.num_psis)
            P4.reactCoord2All[P4.user_PrD].setMaximum(p.num_psis)
        else:
            P4.label_pic8.setPixmap(QtGui.QPixmap(blankpix))
            P4.button_pic8.setDisabled(True)

        EigValCanvas().EigValRead() #read in eigenvalue spectrum for current PrD
        population = len(P1.CG[(P4.user_PrD)-1])
        P4.entry_pop.setValue(population)
        P3.button_toP4.setDisabled(True)
        gotoP4(self)

    ##########
    # Task 1:
    @QtCore.Slot()
    def start_task1(self):
        tabs.setTabEnabled(0, False)
        tabs.setTabEnabled(1, False)

        set_params.op(0) #send new GUI data to user parameters file

        P3.entry_proc.setDisabled(True)
        P3.entry_psi.setDisabled(True)
        P3.entry_dim.setDisabled(True)
        P3.button_dist.setDisabled(True)
        P3.button_dist.setText('Distance Calculation Initiated')
        task1 = threading.Thread(target=GetDistancesS2.op,
                         args=(self.progress1Changed, ))
        task1.daemon = True
        task1.start()

    @QtCore.Slot(int)
    def on_progress1Changed(self, val):
        P3.progress1.setValue(val)
        if val == P3.progress1.maximum():
            P3.button_dist.setText('Distance Calculation Complete')
            P3.button_eig.setDisabled(False)
            gc.collect()
            p.resProj = 2
            set_params.op(0) #send new GUI data to user parameters file
            time.sleep(20) #when distance matrices very large, can take time to store before next module!
            self.start_task2()

    ##########
    # Task 2:
    @QtCore.Slot()
    def start_task2(self):
        tabs.setTabEnabled(0, False)
        tabs.setTabEnabled(1, False)

        set_params.op(0) #send new GUI data to user parameters file
        
        P3.entry_proc.setDisabled(True)
        P3.entry_psi.setDisabled(True)
        P3.entry_dim.setDisabled(True)
        P3.button_eig.setDisabled(True)
        P3.button_eig.setText('Embedding Initiated')
        task2 = threading.Thread(target=manifoldAnalysis.op,
                         args=(self.progress2Changed, ))
        task2.daemon = True
        task2.start()

    @QtCore.Slot(int)
    def on_progress2Changed(self, val):
        P3.progress2.setValue(val)
        if val == P3.progress2.maximum():
            P3.button_eig.setText('Embedding Complete')
            P3.button_psi.setDisabled(False)
            gc.collect()
            p.resProj = 3
            set_params.op(0) #send new GUI data to user parameters file
            time.sleep(5)
            self.start_task3()
            
    ##########
    # Task 3:
    @QtCore.Slot()
    def start_task3(self):
        tabs.setTabEnabled(0, False)
        tabs.setTabEnabled(1, False)

        set_params.op(0) #send new GUI data to user parameters file
        
        P3.entry_proc.setDisabled(True)
        P3.entry_psi.setDisabled(True)
        P3.entry_dim.setDisabled(True)
        P3.button_psi.setDisabled(True)
        P3.button_psi.setText('Spectral Analysis Initiated')
        task3 = threading.Thread(target=psiAnalysis.op,
                         args=(self.progress3Changed, ))
        task3.daemon = True
        task3.start()

    @QtCore.Slot(int)
    def on_progress3Changed(self, val):
        P3.progress3.setValue(val)
        if val == P3.progress3.maximum():
            P3.button_psi.setText('Spectral Analysis Complete')
            gc.collect()
            p.resProj = 4
            set_params.op(0) #send new GUI data to user parameters file
            time.sleep(5)
            self.start_task4()

    ##########
    # Task 4:
    @QtCore.Slot()
    def start_task4(self):
        tabs.setTabEnabled(0, False)
        tabs.setTabEnabled(1, False)

        set_params.op(0) #send new GUI data to user parameters file
        
        P3.entry_proc.setDisabled(True)
        P3.entry_psi.setDisabled(True)
        P3.entry_dim.setDisabled(True)
        P3.button_nlsa.setDisabled(True)
        P3.button_nlsa.setText('Compiling 2D Movies')
        task4 = threading.Thread(target=NLSAmovie.op,
                         args=(self.progress4Changed, ))
        task4.daemon = True
        task4.start()

    @QtCore.Slot(int)
    def on_progress4Changed(self, val):
        P3.progress4.setValue(val)
        if val == P3.progress4.maximum():
            tabs.setTabEnabled(0, True)
            tabs.setTabEnabled(1, True)
            P3.button_nlsa.setText('2D Movies Complete')
            P3.button_toP4.setDisabled(False)
            #P3.entry_proc.setDisabled(False)
            P3.entry_dim.setDisabled(False)
            gc.collect()
            p.resProj = 5
            set_params.op(0) #send new GUI data to user parameters file

                    
# =============================================================================
# GUI tab 4:
# =============================================================================

class P4(QtGui.QWidget):
    user_coord1 = 1
    user_coord2 = 0
    user_azimuth2 = 0
    user_elevation2 = 0
    user_PrD = 1
    PrD_hist = 1 #keeps track of previous PrD visited, for removal
    user_psi = 1
    vidDir = ''
    anch_list = [] #final user anchor selections
    # widget dictionaries for all PrDs:
    anchorsAll = {}
    senses1All = {}
    senses2All = {}
    reactCoord1All = {}
    reactCoord2All = {}
    trash_list = []
    trashAll = {} #user-defined PDs for removal
    origEmbed = [] #keeps track of Manifold Analysis number of re-embeddings
    origEmbedFile = '' #to-file copy of the above (for resumes)
    recompute = 0 #F/T: if Compilation tab has already been computed (0 if not)

    def __init__(self, parent=None):
        super(P4, self).__init__(parent)
        P4.layout = QtGui.QGridLayout(self)
        P4.layout.setContentsMargins(20,20,20,20)
        P4.layout.setSpacing(10)

        P4.layoutL = QtGui.QGridLayout()
        P4.layoutL.setContentsMargins(20,20,20,20)
        P4.layoutL.setSpacing(10)

        P4.layoutR = QtGui.QGridLayout()
        P4.layoutR.setContentsMargins(20,20,20,20)
        P4.layoutR.setSpacing(10)

        P4.layoutB = QtGui.QGridLayout()
        P4.layoutB.setContentsMargins(20,20,20,20)
        P4.layoutB.setSpacing(10)

        self.widgetsL = QtGui.QWidget()
        self.widgetsR = QtGui.QWidget()
        self.widgetsB = QtGui.QWidget()
        self.widgetsL.setLayout(P4.layoutL)
        self.widgetsR.setLayout(P4.layoutR)
        self.widgetsB.setLayout(P4.layoutB)

        global print_anglesP4
        def print_anglesP4(azimuth, elevation):
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s' % progname)
            box.setText('<b>Selected Angles</b>')
            box.setFont(font_standard)
            box.setInformativeText('Azimuth (phi): %s%s<br /><br />Elevation (theta): %s%s< br />'
                                   % (round(azimuth,2), u"\u00B0", round(elevation,2), u"\u00B0"))
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setDefaultButton(QtGui.QMessageBox.Ok)
            ret = box.exec_()

        self.label_topos = QtGui.QLabel("View Topos")
        self.label_topos.setFont(font_standard)
        self.label_topos.setMargin(0)
        self.label_topos.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        P4.layoutR.addWidget(self.label_topos, 0, 8, 1, 4)
        self.label_topos.show()

        # mayavi PrD widget:
        def update_PrD():
            # change angle of 3d plot to correspond with PrD spinbox value:
            P4.user_PrD = P4.entry_PrD.value()
            P4.viz2.update_viewP4(azimuth = float(P3.phi[(P4.user_PrD)-1]),
                                elevation = float(P3.theta[(P4.user_PrD)-1]),
                                distance = P4.viz2.view_anglesP4(dialog = False))

            if MainWindow.restoreComplete == 1:
                P4.viz2.update_euler()

            population = len(P1.CG[(P4.user_PrD)-1])
            P4.entry_pop.setValue(population)

        def update_topos(): #refresh screen for new topos and anchors
            # =================================================================
            # link topos directories and update with photos/videos:
            # =================================================================
            # create blank image if topos file doesn't exist
            # =================================================================
            picDir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_1.png' % (p.proj_name,P4.user_PrD))
            picImg = Image.open(picDir)
            picSize = picImg.size
            blank = np.zeros([picSize[0],picSize[1],3], dtype=np.uint8)
            blank.fill(0)
            blank = QtGui.QImage(blank, blank.shape[1],\
            blank.shape[0], blank.shape[1] * 3,QtGui.QImage.Format_RGB888)
            blankpix = QtGui.QPixmap(blank)
            # =================================================================
            P4.user_PrD = P4.entry_PrD.value()
            topos_sum = 0
            P4.pic1Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_1.png' % (p.proj_name,P4.user_PrD))
            if os.path.isfile(P4.pic1Dir):
                P4.label_pic1.setPixmap(QtGui.QPixmap(P4.pic1Dir))
                P4.button_pic1.setDisabled(False)
                topos_sum += 1
            else:
                P4.label_pic1.setPixmap(QtGui.QPixmap(blankpix))
                P4.button_pic1.setDisabled(True)

            P4.pic2Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_2.png' % (p.proj_name,P4.user_PrD))
            if os.path.isfile(P4.pic2Dir):
                P4.label_pic2.setPixmap(QtGui.QPixmap(P4.pic2Dir))
                P4.button_pic2.setDisabled(False)
                topos_sum += 1
            else:
                P4.label_pic2.setPixmap(QtGui.QPixmap(blankpix))
                P4.button_pic2.setDisabled(True)

            P4.pic3Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_3.png' % (p.proj_name,P4.user_PrD))
            if os.path.isfile(P4.pic3Dir):
                P4.label_pic3.setPixmap(QtGui.QPixmap(P4.pic3Dir))
                P4.button_pic3.setDisabled(False)
                topos_sum += 1
            else:
                P4.label_pic3.setPixmap(QtGui.QPixmap(blankpix))
                P4.button_pic3.setDisabled(True)

            P4.pic4Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_4.png' % (p.proj_name,P4.user_PrD))
            if os.path.isfile(P4.pic4Dir):
                P4.label_pic4.setPixmap(QtGui.QPixmap(P4.pic4Dir))
                P4.button_pic4.setDisabled(False)
                topos_sum += 1
            else:
                P4.label_pic4.setPixmap(QtGui.QPixmap(blankpix))
                P4.button_pic4.setDisabled(True)

            P4.pic5Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_5.png' % (p.proj_name,P4.user_PrD))
            if os.path.isfile(P4.pic5Dir):
                P4.label_pic5.setPixmap(QtGui.QPixmap(P4.pic5Dir))
                P4.button_pic5.setDisabled(False)
                topos_sum += 1
            else:
                P4.label_pic5.setPixmap(QtGui.QPixmap(blankpix))
                P4.button_pic5.setDisabled(True)

            P4.pic6Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_6.png' % (p.proj_name,P4.user_PrD))
            if os.path.isfile(P4.pic6Dir):
                P4.label_pic6.setPixmap(QtGui.QPixmap(P4.pic6Dir))
                P4.button_pic6.setDisabled(False)
                topos_sum += 1
            else:
                P4.label_pic6.setPixmap(QtGui.QPixmap(blankpix))
                P4.button_pic6.setDisabled(True)

            P4.pic7Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_7.png' % (p.proj_name,P4.user_PrD))
            if os.path.isfile(P4.pic7Dir):
                P4.label_pic7.setPixmap(QtGui.QPixmap(P4.pic7Dir))
                P4.button_pic7.setDisabled(False)
                topos_sum += 1
            else:
                P4.label_pic7.setPixmap(QtGui.QPixmap(blankpix))
                P4.button_pic7.setDisabled(True)

            P4.pic8Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_8.png' % (p.proj_name,P4.user_PrD))
            if os.path.isfile(P4.pic8Dir):
                P4.label_pic8.setPixmap(QtGui.QPixmap(P4.pic8Dir))
                P4.button_pic8.setDisabled(False)
                topos_sum += 1
            else:
                P4.label_pic8.setPixmap(QtGui.QPixmap(blankpix))
                P4.button_pic8.setDisabled(True)

            P4.reactCoord1All[P4.user_PrD].setMaximum(topos_sum)
            P4.reactCoord2All[P4.user_PrD].setMaximum(topos_sum)
            EigValCanvas().EigValRead() #read in eigenvalue spectrum for current PrD

            # =================================================================
            # Update trash section widgets:
            # =================================================================
            P4.layoutL.removeWidget(P4.trashAll[P4.PrD_hist])
            P4.trashAll[P4.PrD_hist].close()
            P4.layoutL.addWidget(P4.trashAll[P4.user_PrD], 6, 5, 1, 2, QtCore.Qt.AlignCenter)
            P4.trashAll[P4.user_PrD].show()

            # =================================================================
            # Update anchor section widgets:
            # =================================================================
            if P3.user_dimensions == 1:
                P4.layoutB.removeWidget(P4.reactCoord1All[P4.PrD_hist])
                P4.reactCoord1All[P4.PrD_hist].close()
                P4.layoutB.addWidget(P4.reactCoord1All[P4.user_PrD], 8, 2, 1, 1)
                P4.reactCoord1All[P4.user_PrD].show()

                P4.layoutB.removeWidget(P4.senses1All[P4.PrD_hist])
                P4.senses1All[P4.PrD_hist].close()
                P4.layoutB.addWidget(P4.senses1All[P4.user_PrD], 8, 3, 1, 1)
                P4.senses1All[P4.user_PrD].show()

                P4.layoutB.removeWidget(P4.anchorsAll[P4.PrD_hist])
                P4.anchorsAll[P4.PrD_hist].close()
                P4.layoutB.addWidget(P4.anchorsAll[P4.user_PrD], 8, 4, 1, 1)
                P4.anchorsAll[P4.user_PrD].show()

            if P3.user_dimensions == 2:
                P4.layoutB.removeWidget(P4.reactCoord1All[P4.PrD_hist])
                P4.reactCoord1All[P4.PrD_hist].close()
                P4.layoutB.addWidget(P4.reactCoord1All[P4.user_PrD], 8, 1, 1, 1)
                P4.reactCoord1All[P4.user_PrD].show()

                P4.layoutB.removeWidget(P4.senses1All[P4.PrD_hist])
                P4.senses1All[P4.PrD_hist].close()
                P4.layoutB.addWidget(P4.senses1All[P4.user_PrD], 8, 2, 1, 1)
                P4.senses1All[P4.user_PrD].show()

                P4.layoutB.removeWidget(P4.reactCoord2All[P4.PrD_hist])
                P4.reactCoord2All[P4.PrD_hist].close()
                P4.layoutB.addWidget(P4.reactCoord2All[P4.user_PrD], 8, 3, 1, 1)
                P4.reactCoord2All[P4.user_PrD].show()

                P4.layoutB.removeWidget(P4.senses2All[P4.PrD_hist])
                P4.senses2All[P4.PrD_hist].close()
                P4.layoutB.addWidget(P4.senses2All[P4.user_PrD], 8, 4, 1, 1)
                P4.senses2All[P4.user_PrD].show()

                P4.layoutB.removeWidget(P4.anchorsAll[P4.PrD_hist])
                P4.anchorsAll[P4.PrD_hist].close()
                P4.layoutB.addWidget(P4.anchorsAll[P4.user_PrD], 8, 5, 1, 1)
                P4.anchorsAll[P4.user_PrD].show()

            # add current PrD to history list:
            P4.PrD_hist = P4.user_PrD

        P4.viz2 = Mayavi_Rho() #P4.viz2 = Mayavi_Rho(PrD_high = 2)
        P4.ui2 = P4.viz2.edit_traits(parent=self, kind='subpanel').control
        P4.layoutL.addWidget(P4.ui2, 0, 0, 6, 7)

        self.label_PrD = QtGui.QLabel('Projection Direction:')
        self.label_PrD.setFont(font_header)
        self.label_PrD.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        P4.layoutL.addWidget(self.label_PrD, 6, 0, 1, 1)
        self.label_PrD.show()

        P4.entry_PrD = QtGui.QSpinBox(self)
        P4.entry_PrD.setMinimum(1)
        P4.entry_PrD.setMaximum(P3.PrD_total)
        P4.entry_PrD.setSuffix('  /  %s' % (P3.PrD_total))
        P4.entry_PrD.valueChanged.connect(update_PrD)
        P4.entry_PrD.valueChanged.connect(update_topos)
        P4.entry_PrD.setToolTip('Change the projection direction of the current view above.')
        P4.layoutL.addWidget(P4.entry_PrD, 6, 1, 1, 2)
        P4.entry_PrD.show()

        P4.entry_pop = QtGui.QDoubleSpinBox(self)
        P4.entry_pop.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)
        P4.entry_pop.setToolTip('Total number of particles within current PD.')
        P4.entry_pop.setDisabled(True)
        P4.entry_pop.setDecimals(0)
        P4.entry_pop.setMaximum(99999999)
        P4.entry_pop.setSuffix(' images')
        P4.layoutL.addWidget(P4.entry_pop, 6, 3, 1, 2)
        P4.entry_pop.show()

        # conformational coordinates:
        minSize = 1
                                
        def eigSpectrum():
            global EigValMain_window
            try:
                EigValMain_window.close()
            except:
                pass
            EigValMain_window = EigValMain()
            EigValMain_window.setMinimumSize(10, 10)
            EigValMain_window.setWindowTitle('Projection Direction %s' % (P4.user_PrD))
            EigValMain_window.show()

        def classAvg():
            global ClassAvgMain_window
            try:
                ClassAvgMain_window.close()
            except:
                pass
            ClassAvgMain_window = ClassAvgMain()
            ClassAvgMain_window.setMinimumSize(10, 10)
            ClassAvgMain_window.setWindowTitle('Projection Direction %s' % (P4.user_PrD))
            ClassAvgMain_window.show()
            
        P4.label_pic1 = QtGui.QLabel()
        P4.pic1Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_1.png' % (p.proj_name,P4.user_PrD))
        P4.label_pic1.setPixmap(QtGui.QPixmap(P4.pic1Dir))
        P4.label_pic1.setMinimumSize(minSize, minSize)
        P4.label_pic1.setScaledContents(True)
        P4.label_pic1.setAlignment(QtCore.Qt.AlignCenter)
        P4.layoutR.addWidget(P4.label_pic1, 1, 8, 1, 1)
        P4.label_pic1.show()
        
        P4.button_pic1 = QtGui.QPushButton('View %s%s' % (u"\u03A8",u"\u2081"), self)
        P4.button_pic1.clicked.connect(lambda: self.CC_vid1(1))
        P4.button_pic1.setToolTip('View 2d movie and related outputs.')
        P4.layoutR.addWidget(P4.button_pic1, 2, 8, 1, 1)
        P4.button_pic1.show()

        P4.label_pic2 = QtGui.QLabel()
        P4.pic2Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_2' % (p.proj_name,P4.user_PrD))
        P4.label_pic2.setPixmap(QtGui.QPixmap(P4.pic2Dir))
        P4.label_pic2.setMinimumSize(minSize, minSize)
        P4.label_pic2.setScaledContents(True)
        P4.label_pic2.setAlignment(QtCore.Qt.AlignCenter)
        P4.layoutR.addWidget(P4.label_pic2, 1, 9, 1, 1)
        P4.label_pic2.show()

        P4.button_pic2 = QtGui.QPushButton('View %s%s' % (u"\u03A8",u"\u2082"), self)
        P4.button_pic2.clicked.connect(lambda: self.CC_vid1(2))
        P4.button_pic2.setToolTip('View 2d movie and related outputs.')
        P4.layoutR.addWidget(P4.button_pic2, 2, 9, 1, 1)
        P4.button_pic2.show()

        P4.label_pic3 = QtGui.QLabel()
        P4.pic3Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_3' % (p.proj_name,P4.user_PrD))
        P4.label_pic3.setPixmap(QtGui.QPixmap(P4.pic3Dir))
        P4.label_pic3.setMinimumSize(minSize, minSize)
        P4.label_pic3.setScaledContents(True)
        P4.label_pic3.setAlignment(QtCore.Qt.AlignCenter)
        P4.layoutR.addWidget(P4.label_pic3, 1, 10, 1, 1)
        P4.label_pic3.show()

        P4.button_pic3 = QtGui.QPushButton('View %s%s' % (u"\u03A8",u"\u2083"), self)
        P4.button_pic3.clicked.connect(lambda: self.CC_vid1(3))
        P4.button_pic3.setToolTip('View 2d movie and related outputs.')
        P4.layoutR.addWidget(P4.button_pic3, 2, 10, 1, 1)
        P4.button_pic3.show()

        P4.label_pic4 = QtGui.QLabel()
        P4.pic4Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_4' % (p.proj_name,P4.user_PrD))
        P4.label_pic4.setPixmap(QtGui.QPixmap(P4.pic4Dir))
        P4.label_pic4.setMinimumSize(minSize, minSize)
        P4.label_pic4.setScaledContents(True)
        P4.label_pic4.setAlignment(QtCore.Qt.AlignCenter)
        P4.layoutR.addWidget(P4.label_pic4, 1, 11, 1, 1)
        P4.label_pic4.show()

        P4.button_pic4 = QtGui.QPushButton('View %s%s' % (u"\u03A8",u"\u2084"), self)
        P4.button_pic4.clicked.connect(lambda: self.CC_vid1(4))
        P4.button_pic4.setToolTip('View 2d movie and related outputs.')
        P4.layoutR.addWidget(P4.button_pic4, 2, 11, 1, 1)
        P4.button_pic4.show()

        self.label_Hline = QtGui.QLabel("")
        self.label_Hline.setFont(font_standard)
        self.label_Hline.setMargin(0)
        self.label_Hline.setFrameStyle(QtGui.QFrame.HLine | QtGui.QFrame.Sunken)
        P4.layoutR.addWidget(self.label_Hline, 3, 8, 1, 4)
        self.label_Hline.show() 

        P4.label_pic5 = QtGui.QLabel()
        P4.pic5Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_5' % (p.proj_name,P4.user_PrD))
        P4.label_pic5.setPixmap(QtGui.QPixmap(P4.pic5Dir))
        P4.label_pic5.setMinimumSize(minSize, minSize)
        P4.label_pic5.setScaledContents(True)
        P4.label_pic5.setAlignment(QtCore.Qt.AlignCenter)
        P4.layoutR.addWidget(P4.label_pic5, 4, 8, 1, 1)
        P4.label_pic5.show()
        
        P4.button_pic5 = QtGui.QPushButton('View %s%s' % (u"\u03A8",u"\u2085"), self)
        P4.button_pic5.clicked.connect(lambda: self.CC_vid1(5))
        P4.button_pic5.setToolTip('View 2d movie and related outputs.')
        P4.layoutR.addWidget(P4.button_pic5, 5, 8, 1, 1)
        P4.button_pic5.show()

        P4.label_pic6 = QtGui.QLabel()
        P4.pic6Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_6' % (p.proj_name,P4.user_PrD))
        P4.label_pic6.setPixmap(QtGui.QPixmap(P4.pic6Dir))
        P4.label_pic6.setMinimumSize(minSize, minSize)
        P4.label_pic6.setScaledContents(True)
        P4.label_pic6.setAlignment(QtCore.Qt.AlignCenter)
        P4.layoutR.addWidget(P4.label_pic6, 4, 9, 1, 1)
        P4.label_pic6.show()
        
        P4.button_pic6 = QtGui.QPushButton('View %s%s' % (u"\u03A8",u"\u2086"), self)
        P4.button_pic6.clicked.connect(lambda: self.CC_vid1(6))
        P4.button_pic6.setToolTip('View 2d movie and related outputs.')
        P4.layoutR.addWidget(P4.button_pic6, 5, 9, 1, 1)
        P4.button_pic6.show()

        P4.label_pic7 = QtGui.QLabel()
        P4.pic7Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_7' % (p.proj_name,P4.user_PrD))
        P4.label_pic7.setPixmap(QtGui.QPixmap(P4.pic7Dir))
        P4.label_pic7.setMinimumSize(minSize, minSize)
        P4.label_pic7.setScaledContents(True)
        P4.label_pic7.setAlignment(QtCore.Qt.AlignCenter)
        P4.layoutR.addWidget(P4.label_pic7, 4, 10, 1, 1)
        P4.label_pic7.show()
        
        P4.button_pic7 = QtGui.QPushButton('View %s%s' % (u"\u03A8",u"\u2087"), self)
        P4.button_pic7.clicked.connect(lambda: self.CC_vid1(7))
        P4.button_pic7.setToolTip('View 2d movie and related outputs.')
        P4.layoutR.addWidget(P4.button_pic7, 5, 10, 1, 1)
        P4.button_pic7.show()

        P4.label_pic8 = QtGui.QLabel()
        P4.pic8Dir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_8' % (p.proj_name,P4.user_PrD))
        P4.label_pic8.setPixmap(QtGui.QPixmap(P4.pic8Dir))
        P4.label_pic8.setMinimumSize(minSize, minSize)
        P4.label_pic8.setScaledContents(True)
        P4.label_pic8.setAlignment(QtCore.Qt.AlignCenter)
        P4.layoutR.addWidget(P4.label_pic8, 4, 11, 1, 1)
        P4.label_pic8.show()
        
        P4.button_pic8 = QtGui.QPushButton('View %s%s' % (u"\u03A8",u"\u2088"), self)
        P4.button_pic8.clicked.connect(lambda: self.CC_vid1(8))
        P4.button_pic8.setToolTip('View 2d movie and related outputs.')
        P4.layoutR.addWidget(P4.button_pic8, 5, 11, 1, 1)
        P4.button_pic8.show()

        P4.button_bandwidth = QtGui.QPushButton('Kernel Bandwidth')
        P4.button_bandwidth.setDisabled(False)
        self.button_bandwidth.clicked.connect(self.bandwidth)
        P4.layoutR.addWidget(P4.button_bandwidth, 6, 8, 1, 1)
        P4.button_bandwidth.show()

        P4.button_eigSpec = QtGui.QPushButton('Eigenvalue Spectrum')
        self.button_eigSpec.clicked.connect(eigSpectrum)
        P4.layoutR.addWidget(P4.button_eigSpec, 6, 9, 1, 1)
        P4.button_eigSpec.show()

        P4.button_viewAvg = QtGui.QPushButton('2D Class Average')
        self.button_viewAvg.clicked.connect(classAvg)
        P4.layoutR.addWidget(P4.button_viewAvg, 6, 10, 1, 1)
        P4.button_viewAvg.show()
        
        P4.button_compareMov = QtGui.QPushButton('Compare Movies')
        self.button_compareMov.clicked.connect(self.CC_vid2)
        P4.layoutR.addWidget(P4.button_compareMov, 6, 11, 1, 1)
        P4.button_compareMov.show()

        if 0:
            maxSize = 300
            self.label_pic1.setMaximumSize(maxSize, maxSize)
            self.label_pic2.setMaximumSize(maxSize, maxSize)
            self.label_pic3.setMaximumSize(maxSize, maxSize)
            self.label_pic4.setMaximumSize(maxSize, maxSize)
            self.label_pic5.setMaximumSize(maxSize, maxSize)
            self.label_pic6.setMaximumSize(maxSize, maxSize)
            self.label_pic7.setMaximumSize(maxSize, maxSize)
            self.label_pic8.setMaximumSize(maxSize, maxSize)

        # confirm conformational coordinates:
        def anchorCheck():
            # first save PDs chosen for removal to file:
            P4.trash_list = []
            for i in range(1,P3.PrD_total+1):
                if P4.trashAll[i].isChecked():
                    P4.trash_list.append(int(1))
                else:
                    P4.trash_list.append(int(0))

            #### ZULU ####
            #{SECTION EVENTUALLY NEEDED FOR RE-DOING GRAPH IF TRASH LIST DIVIDED ANY ISLANDS INTO SUB-ISLANDS}
            
            #if len(P4.trash_list) > 0:
                #trash_reviewed = T or F variable
            ##############

            trashDir = os.path.join(P1.user_directory, 'outputs_{}/CC/user_removals.txt'.format(p.proj_name))
            np.savetxt(trashDir, P4.trash_list, fmt='%i', delimiter='\t')
            p.trash_list = P4.trash_list

            # save anchors to file:
            anch_sum = 0
            for i in range(1,P3.PrD_total+1):
                if P4.anchorsAll[i].isChecked():
                    anch_sum += 1
            if anch_sum < anchorsMin:    
                box = QtGui.QMessageBox(self)
                box.setWindowTitle('%s Error' % progname)
                box.setText('<b>Input Error</b>')
                box.setFont(font_standard)
                box.setIcon(QtGui.QMessageBox.Information)
                box.setInformativeText('A minimum of %s PD anchors must be selected.' % anchorsMin)
                box.setStandardButtons(QtGui.QMessageBox.Ok)
                box.setDefaultButton(QtGui.QMessageBox.Ok)
                ret = box.exec_()

            elif anch_sum >= anchorsMin:
                PrDs = []
                CC1s = []
                S1s = []
                CC2s = []
                S2s = []
                colors = []
                P4.anch_list = []

                idx = 0
                for i in range(1,P3.PrD_total+1):
                    if P4.anchorsAll[i].isChecked():
                        PrDs.append(int(i))
                        # CC1s:
                        CC1s.append(int(P4.reactCoord1All[i].value()))
                        # S1s:
                        if P4.senses1All[i].currentText() == 'S1: FWD':
                            S1s.append(int(1))
                        else:
                            S1s.append(int(-1))
                        # CC2s:
                        CC2s.append(int(P4.reactCoord2All[i].value()))
                        # S2s:
                        if P4.senses2All[i].currentText() == 'S2: FWD':
                            S2s.append(int(1))
                        else:
                            S2s.append(int(-1))
                        # colors:
                        colors.append(P3.col[int(i-1)])
                        idx += 1

                if P3.user_dimensions == 1:
                    P4.anch_list = zip(PrDs,CC1s,S1s,colors)
                elif P3.user_dimensions == 2:
                    P4.anch_list = zip(PrDs,CC1s,S1s,CC2s,S2s,colors)

                # check if at least one anchor is selected for each color:
                if sorted(np.unique(P3.col)) == sorted(np.unique(colors)):
                    msg = 'Performing this action will initiate Belief Propagation for the current \
                            PD anchors and generate the corresponding energy landscape and 3D volumes.\
                            <br /><br />\
                            Do you want to proceed?' 
                    box = QtGui.QMessageBox(self)
                    box.setWindowTitle('%s' % progname)
                    box.setText('<b>Confirm Conformational Coordinates</b>')
                    box.setFont(font_standard)
                    box.setIcon(QtGui.QMessageBox.Question)
                    box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
                    box.setInformativeText(msg)
                    reply = box.exec_()
                    if reply == QtGui.QMessageBox.Yes:
                        P4.anch_list, anch_zip = itertools.tee(P4.anch_list)
                        p.anch_list = list(anch_zip) #PrD,CC1,S1 for 1D; PrD,CC1,S1,CC2,S2 for 2D
                        anchInputs = os.path.join(P1.user_directory, 'outputs_{}/CC/user_anchors.txt'.format(p.proj_name))
                        np.savetxt(anchInputs, p.anch_list, fmt='%i', delimiter='\t')
                        gotoP5(self)
                        if P4.recompute == 0:
                            P4.btn_finOut.setText('Recompile Results')
                            P4.recompute = 1
                            p.resProj = 6  # update progress
                            set_params.op(0)  # send new GUI data to user parameters file
                        else: #overwrite warning
                            msg = 'Final outputs have already been computed for a previous\
                                    selection of anchor nodes. To recompute final outputs\
                                    with new anchor node settings, previous outputs must be\
                                    overwritten.\
                                    <br /><br />\
                                    Do you want to proceed?'
                            box = QtGui.QMessageBox(self)
                            box.setWindowTitle('%s Warning' % progname)
                            box.setText('<b>Overwrite Warning</b>')
                            box.setIcon(QtGui.QMessageBox.Warning)
                            box.setFont(font_standard)
                            box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
                            box.setInformativeText(msg)
                            reply = box.exec_()
                            if reply == QtGui.QMessageBox.Yes: 
                                P5.progress5.setValue(0)
                                P5.progress6.setValue(0)
                                Erg1dMain.progress7.setValue(0)
                                P5.button_CC.setText('Find Conformational Coordinates')
                                P5.button_erg.setText('Energy Landscape')
                                Erg1dMain.button_traj.setText('Compute 3D Trajectories')
                                Erg1dMain.reprepare = 0
                                P5.entry_opt.setDisabled(False)
                                P5.entry_proc.setDisabled(False)
                                P5.entry_temp.setDisabled(False)
                                P5.button_CC.setDisabled(False)
                                P5.button_erg.setDisabled(True)
                                Erg1dMain.button_traj.setDisabled(True)
                                P5.button_toP6.setDisabled(True)
                                tabs.setTabEnabled(5, False)

                                p.resProj = 6 #revert progress back to before ever running FindConformationalCoords.py
                                set_params.op(0) #send new GUI data to user parameters file

                                # =============================================
                                # Hard-remove pre-existing folders:
                                shutil.rmtree(p.CC_meas_dir)
                                shutil.rmtree(p.CC_OF_dir)
                                shutil.rmtree(p.EL_dir)
                                prepOutDir = os.path.join(p.out_dir,'bin')
                                shutil.rmtree(prepOutDir)
                                shutil.rmtree(p.traj_dir)

                                time.sleep(1)
                                os.makedirs(p.CC_meas_dir)
                                os.makedirs(p.CC_OF_dir)
                                os.makedirs(p.EL_dir)
                                os.makedirs(p.OM_dir)
                                os.makedirs(prepOutDir)
                                os.makedirs(p.traj_dir)
                                os.makedirs(p.CC_meas_prog) #progress bar folder
                                os.makedirs(p.EL_prog) #progress bar folder
                                # =============================================
                            else:
                                pass
                    else:
                        pass
                else:
                    msg = 'It is highly recommended that at least one anchor node is selected\
                            for each connected component (as seen via clusters of colored PDs on S2).\
                            <br /><br />\
                            Currently, only %s of %s connected components are satisfied in this manner,\
                            and thus, %s will be ignored during Belief Propagation.\
                            <br /><br />\
                            Do you want to proceed?' % (len(sorted(np.unique(colors))),
                                                        len(sorted(np.unique(P3.col))),
                                                        len(sorted(np.unique(P3.col))) - len(sorted(np.unique(colors)))) 
                    box = QtGui.QMessageBox(self)
                    box.setWindowTitle('%s Warning' % progname)
                    box.setIcon(QtGui.QMessageBox.Warning)
                    box.setText('<b>Input Warning</b>')
                    box.setFont(font_standard)
                    box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
                    box.setInformativeText(msg)
                    reply = box.exec_()
                    if reply == QtGui.QMessageBox.Yes:
                        P4.anch_list, anch_zip = itertools.tee(P4.anch_list)
                        p.anch_list = list(anch_zip) #PrD,CC1,S1 for 1D; PrD,CC1,S1,CC2,S2 for 2D
                        anchInputs = os.path.join(P1.user_directory, 'outputs_{}/CC/user_anchors.txt'.format(p.proj_name))
                        np.savetxt(anchInputs, p.anch_list, fmt='%i', delimiter='\t')
                        gotoP5(self)
                        if P4.recompute == 0:
                            P4.btn_finOut.setText('Recompile Results')
                            P4.recompute = 1
                            p.resProj = 6  #update progress
                            set_params.op(0)  #send new GUI data to user parameters file
                        else: #overwrite warning
                            msg = 'Final outputs have already been computed for a previous\
                                    selection of anchor nodes. To recompute final outputs\
                                    with new anchor node settings, previous outputs must be\
                                    overwritten.\
                                    <br /><br />\
                                    Do you want to proceed?'
                            box = QtGui.QMessageBox(self)
                            box.setWindowTitle('%s Warning' % progname)
                            box.setIcon(QtGui.QMessageBox.Warning)
                            box.setText('<b>Overwrite Warning</b>')
                            box.setFont(font_standard)
                            box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
                            box.setInformativeText(msg)
                            reply = box.exec_()
                            if reply == QtGui.QMessageBox.Yes:
                                P5.progress5.setValue(0)
                                P5.progress6.setValue(0)
                                Erg1dMain.progress7.setValue(0)
                                P5.button_CC.setText('Find Conformational Coordinates')
                                P5.button_erg.setText('Energy Landscape')
                                Erg1dMain.button_traj.setText('Compute 3D Trajectories')
                                Erg1dMain.reprepare = 0
                                P5.entry_opt.setDisabled(False)
                                P5.entry_proc.setDisabled(False)
                                P5.entry_temp.setDisabled(False)
                                P5.button_CC.setDisabled(False)
                                P5.button_erg.setDisabled(True)
                                Erg1dMain.button_traj.setDisabled(True)
                                P5.button_toP6.setDisabled(True)
                                tabs.setTabEnabled(5, False)

                                p.resProj = 6 #revert progress back to before ever running FindConformationalCoords.py
                                set_params.op(0) #send new GUI data to user parameters file
                                    
                                # =============================================
                                # Hard-remove pre-existing folders:
                                shutil.rmtree(p.CC_meas_dir)
                                shutil.rmtree(p.CC_OF_dir)
                                shutil.rmtree(p.EL_dir)
                                prepOutDir = os.path.join(p.out_dir,'bin')
                                shutil.rmtree(prepOutDir)
                                shutil.rmtree(p.traj_dir)

                                time.sleep(1)
                                os.makedirs(p.CC_meas_dir)
                                os.makedirs(p.CC_OF_dir)
                                os.makedirs(p.EL_dir)
                                os.makedirs(p.OM_dir)
                                os.makedirs(prepOutDir)
                                os.makedirs(p.traj_dir)
                                os.makedirs(p.CC_meas_prog) #progress bar folder
                                os.makedirs(p.EL_prog) #progress bar folder
                                # =============================================
                                
                            else:
                                pass
                    else:
                        pass

        self.label_edgeAnchor = QtGui.QLabel('')
        self.label_edgeAnchor.setMargin(5)
        self.label_edgeAnchor.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        P4.layoutB.addWidget(self.label_edgeAnchor, 7, 0, 3, 7)
        self.label_edgeAnchor.show()

        self.label_anchor = QtGui.QLabel('Set PD Anchors')
        self.label_anchor.setFont(font_header)
        self.label_anchor.setMargin(5)
        self.label_anchor.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        self.label_anchor.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        P4.layoutB.addWidget(self.label_anchor, 7, 0, 1, 7)
        self.label_anchor.show()

        self.label_edgeCC = QtGui.QLabel('')
        self.label_edgeCC.setMargin(5)
        self.label_edgeCC.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        P4.layoutB.addWidget(self.label_edgeCC, 7, 8, 3, 4)
        self.label_edgeCC.show()
        
        self.label_reactCoord = QtGui.QLabel('Confirm Conformational Coordinates')
        self.label_reactCoord.setFont(font_header)
        self.label_reactCoord.setMargin(5)
        self.label_reactCoord.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        self.label_reactCoord.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        P4.layoutB.addWidget(self.label_reactCoord, 7, 8, 1, 4)
        self.label_reactCoord.show()

        self.btn_PDsele = QtGui.QPushButton('   PD Selections   ', self)
        self.btn_PDsele.setToolTip('Review current PD selections.')
        self.btn_PDsele.clicked.connect(self.PDSeleViz)
        self.btn_PDsele.setDisabled(False)
        P4.layoutB.addWidget(self.btn_PDsele, 8, 9, 1, 1, QtCore.Qt.AlignCenter)
        self.btn_PDsele.show()

        P4.btn_finOut = QtGui.QPushButton('   Compile Results   ', self)
        P4.btn_finOut.setToolTip('Proceed to next section.')
        P4.btn_finOut.clicked.connect(anchorCheck)
        P4.btn_finOut.setDisabled(False)
        P4.layoutB.addWidget(P4.btn_finOut, 8, 10, 1, 1, QtCore.Qt.AlignLeft)
        P4.btn_finOut.show()

        # layout dividers:
        self.splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.splitter1.addWidget(self.widgetsL)
        self.splitter1.addWidget(self.widgetsR)
        self.splitter1.setStretchFactor(1,1)

        self.splitter2 = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.splitter2.addWidget(self.splitter1)
        self.splitter2.addWidget(self.widgetsB)

        P4.layout.addWidget(self.splitter2)

    def PDSeleViz(self):
        global PDSele_window
        try:
            PDSele_window.close()
        except:
            pass
        PDSele_window = PDSeleMain()
        PDSele_window.setMinimumSize(10, 10)
        PDSele_window.setWindowTitle('Projection Direction Selections')
        PDSele_window.show()

    def on_button(self,n):
        print('Button {0} clicked'.format(n))

    def bandwidth(self):
        global BandwidthMain_window
        try:
            BandwidthMain_window.close()
        except:
            pass
        BandwidthMain_window = BandwidthMain()
        BandwidthMain_window.setMinimumSize(10, 10)
        BandwidthMain_window.setWindowTitle('Projection Direction %s' % (P4.user_PrD))
        BandwidthMain_window.show()

    def CC_vid1(self,n):
        P4.vidDir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/psi_%s/' % (p.proj_name,P4.user_PrD,n))
        global PrD_window
        try:
            PrD_window.close()
        except:
            pass

        Manifold2dCanvas.coordsX = []
        Manifold2dCanvas.coordsY = []
        Manifold2dCanvas.eig_current = n
        eig_n_others = []
        eig_v_others = []
        index = 0
        for i in EigValCanvas.eig_v: #find next highest eigenvalue
            index += 1
            if index != n:
                eig_n_others.append(EigValCanvas.eig_n[index-1])
                eig_v_others.append(EigValCanvas.eig_v[index-1])

        Manifold2dCanvas.eig_compare1 = eig_n_others[0] #max eigenvalue (other than one selected)
        Manifold2dCanvas.eig_compare2 = eig_n_others[1] #next highest eigenvalue from the above

        p.eig_current = Manifold2dCanvas.eig_current
        p.eig_compare1 = Manifold2dCanvas.eig_compare1

        VidCanvas.run = 0
        VidCanvas.img_paths = []
        VidCanvas.imgs = []
        VidCanvas.frames = 0
        PrD_window = PrD_Viz()
        PrD_window.setWindowTitle('PD %s: Psi %s' % (P4.user_PrD, n))
        PrD_window.show()

    def CC_vid2(self):
        global PrD2_window
        try:
            PrD2_window.close()
        except:
            pass

        Vid2Canvas.run = 0
        Vid2Canvas.vidDir1 = ''
        Vid2Canvas.vidDir2 = ''
        Vid2Canvas.imgDir1 = ''
        Vid2Canvas.imgDir2 = ''
        Vid2Canvas.img_paths = []
        Vid2Canvas.img_paths = []
        Vid2Canvas.imgs1 = []
        Vid2Canvas.imgs2 = []
        Vid2Canvas.frames1 = 0
        Vid2Canvas.frames2 = 0
        PrD2_window = PrD2_Viz()

        PrD2_window.setWindowTitle('Compare NLSA Movies')
        PrD2_window.show()
        

class anchTable(QtGui.QTableWidget):
    def __init__(self, parent=None, *args, **kwds):
        QtGui.QTableWidget.__init__(self, parent)
        self.library_values = kwds['data']
        self.BuildTable(self.library_values)
        
    def AddToTable(self, values):
        for k,  v in enumerate(values):
            self.AddItem(k, v)
            
    def AddItem(self, row, data):
        for column, value in enumerate(data):
            item = QtGui.QTableWidgetItem(value)
            item = QtGui.QTableWidgetItem(str(value))
            self.setItem(row, column, item)

    def BuildTable(self, values):
        self.setSortingEnabled(False)
        if P3.user_dimensions == 1:
            headers = ['PD','CC','Sense','Color']
        elif P3.user_dimensions == 2:
            headers = ['PD','CC1','Sense1','CC2','Sense2','Color']

        valuesList = list(values)
        self.setRowCount(len(valuesList))
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setWindowTitle('Review PD Anchors')
        self.AddToTable(valuesList)
        self.resizeColumnsToContents()

class occTable(QtGui.QTableWidget):
    def __init__(self, parent=None, *args, **kwds):
        QtGui.QTableWidget.__init__(self, parent)
        self.library_values = kwds['data']
        self.BuildTable(self.library_values)

    def AddToTable(self, values):
        for k,  v in enumerate(values):
            self.AddItem(k, v)

    def AddItem(self, row, data):
        for column, value in enumerate(data):
            item = QtGui.QTableWidgetItem(value)
            item = QtGui.QTableWidgetItem(str(value))
            self.setItem(row, column, item)

    def BuildTable(self, values):
        self.setSortingEnabled(False)
        headers = ['PD','Occ']
        self.setRowCount(len(values))
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setWindowTitle('PD Occupancies')
        self.AddToTable(values)
        self.resizeColumnsToContents()

class trashTable(QtGui.QTableWidget):
    def __init__(self, parent=None, *args, **kwds):
        QtGui.QTableWidget.__init__(self, parent)
        self.library_values = kwds['data']
        self.BuildTable(self.library_values)

    def AddToTable(self, values):
        for k,  v in enumerate(values):
            self.AddItem(k, v)

    def AddItem(self, row, data):
        for column, value in enumerate(data):
            item = QtGui.QTableWidgetItem(value)
            item = QtGui.QTableWidgetItem(str(value))
            self.setItem(row, column, item)

    def BuildTable(self, values):
        self.setSortingEnabled(False)
        headers = ['PD', 'Removed']
        self.setRowCount(len(values))
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setWindowTitle('PD Removals')
        self.AddToTable(values)
        self.resizeColumnsToContents()

class rebedTable(QtGui.QTableWidget):
    def __init__(self, parent=None, *args, **kwds):
        QtGui.QTableWidget.__init__(self, parent)
        self.library_values = kwds['data']
        self.BuildTable(self.library_values)

    def AddToTable(self, values):
        for k,  v in enumerate(values):
            self.AddItem(k, v)

    def AddItem(self, row, data):
        for column, value in enumerate(data):
            item = QtGui.QTableWidgetItem(value)
            item = QtGui.QTableWidgetItem(str(value))
            self.setItem(row, column, item)

    def BuildTable(self, values):
        self.setSortingEnabled(False)
        headers = ['PD', 'Re-embedded']
        self.setRowCount(len(values))
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setWindowTitle('PD Re-embeddings')
        self.AddToTable(values)
        self.resizeColumnsToContents()

# =============================================================================
# GUI page 5:
# =============================================================================

class P5(QtGui.QWidget):
    #temporary values:
    user_temperature = 25 #Celsius
    user_optFlow = 0

    # threading:
    progress5Changed = QtCore.Signal(int)
    progress6Changed = QtCore.Signal(int)

    def __init__(self, parent=None):
        super(P5, self).__init__(parent)
        layout = QtGui.QGridLayout(self)
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(10)

        # =====================================================================
        # inputs section:
        # =====================================================================
        def choose_processors():
            P3.user_processors = P5.entry_proc.value()
            p.ncpu = P5.entry_proc.value()
            P3.entry_proc.setValue(P5.entry_proc.value())

        def choose_temperature():
            P5.user_temperature = self.entry_temp.value()
            p.temperature = P5.user_temperature

        def choose_optFlow():
            if P5.entry_opt.currentText() == 'Yes':
                P5.user_optFlow = int(1)
            else:
                P5.user_optFlow = int(0)
            p.opt_movie["printFig"] = P5.user_optFlow

        # forced space top:
        self.label_spaceT = QtGui.QLabel("")
        self.label_spaceT.setFont(font_standard)
        self.label_spaceT.setMargin(0)
        layout.addWidget(self.label_spaceT, 0, 0, 1, 7, QtCore.Qt.AlignVCenter)
        self.label_spaceT.show()

        # main outline:
        self.label_edgeMain = QtGui.QLabel('')
        self.label_edgeMain.setMargin(20)
        self.label_edgeMain.setLineWidth(1)
        self.label_edgeMain.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edgeMain, 0, 0, 13, 8)
        self.label_edgeMain.show()

        self.label_edge0 = QtGui.QLabel('')
        self.label_edge0.setMargin(20)
        self.label_edge0.setLineWidth(1)
        self.label_edge0.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge0, 1, 1, 1, 2)
        self.label_edge0.show()

        self.label_edge0a = QtGui.QLabel('')
        self.label_edge0a.setMargin(20)
        self.label_edge0a.setLineWidth(1)
        self.label_edge0a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge0a, 1, 1, 1, 1)
        self.label_edge0a.show()

        self.label_proc = QtGui.QLabel('Processors')
        self.label_proc.setFont(font_standard)
        self.label_proc.setMargin(20)
        self.label_proc.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        self.label_proc.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_proc, 1, 1, 1, 1)
        self.label_proc.show()

        P5.entry_proc = QtGui.QSpinBox(self)
        P5.entry_proc.setMinimum(1)
        if p.machinefile:
            P5.entry_proc.setMaximum(1000)
        else:
            P5.entry_proc.setMaximum(multiprocessing.cpu_count())
        P5.entry_proc.valueChanged.connect(choose_processors)
        P5.entry_proc.setStyleSheet("QSpinBox { width : 100px }")
        layout.addWidget(P5.entry_proc, 1, 2, 1, 1, QtCore.Qt.AlignLeft)
        P5.entry_proc.setToolTip('The number of processors to use in parallel.')
        P5.entry_proc.show()

        self.label_edge00 = QtGui.QLabel('')
        self.label_edge00.setMargin(20)
        self.label_edge00.setLineWidth(1)
        self.label_edge00.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge00, 1, 3, 1, 2)
        self.label_edge00.show()
        
        self.label_edge00a = QtGui.QLabel('')
        self.label_edge00a.setMargin(20)
        self.label_edge00a.setLineWidth(1)
        self.label_edge00a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge00a, 1, 3, 1, 1)
        self.label_edge00a.show()

        self.label_temp = QtGui.QLabel('Temperature')
        self.label_temp.setFont(font_standard)
        self.label_temp.setMargin(20)
        self.label_temp.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        self.label_temp.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_temp, 1, 3, 1, 1)
        self.label_temp.show()

        P5.entry_temp = QtGui.QSpinBox(self)
        P5.entry_temp.setMinimum(0)
        P5.entry_temp.setMaximum(100)
        P5.entry_temp.setValue(25)
        P5.entry_temp.setSuffix(' C')
        P5.entry_temp.valueChanged.connect(choose_temperature)
        P5.entry_temp.setStyleSheet("QSpinBox { width : 100px }")
        P5.entry_temp.setToolTip('The temperature of the sample prior to quenching.')
        layout.addWidget(P5.entry_temp, 1, 4, 1, 1, QtCore.Qt.AlignLeft)
        P5.entry_temp.show()

        self.label_edge000 = QtGui.QLabel('')
        self.label_edge000.setMargin(20)
        self.label_edge000.setLineWidth(1)
        self.label_edge000.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge000, 1, 5, 1, 2)
        self.label_edge000.show()
        
        self.label_edge000a = QtGui.QLabel('')
        self.label_edge000a.setMargin(20)
        self.label_edge000a.setLineWidth(1)
        self.label_edge000a.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge000a, 1, 5, 1, 1)
        self.label_edge000a.show()

        self.label_opt = QtGui.QLabel('Export Optical Flow')
        self.label_opt.setFont(font_standard)
        self.label_opt.setMargin(20)
        self.label_opt.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        self.label_opt.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_opt, 1, 5, 1, 1)
        self.label_opt.show()

        P5.entry_opt = QtGui.QComboBox(self)
        P5.entry_opt.setFont(font_standard)
        P5.entry_opt.addItem('No')
        P5.entry_opt.addItem('Yes')
        P5.entry_opt.setToolTip('Export flow vector images to outputs directory.')
        P5.entry_opt.currentIndexChanged.connect(choose_optFlow)
        layout.addWidget(P5.entry_opt, 1, 6, 1, 1, QtCore.Qt.AlignLeft)
        P5.entry_opt.show()

        # conformational coordinates progress:
        P5.button_CC = QtGui.QPushButton('Find Conformational Coordinates', self)
        P5.button_CC.clicked.connect(self.start_task5)
        layout.addWidget(P5.button_CC, 3, 1, 1, 2)
        P5.button_CC.setDisabled(False)
        P5.button_CC.show()

        P5.progress5 = QtGui.QProgressBar(minimum=0, maximum=100, value=0)
        self.progress5Changed.connect(self.on_progress5Changed)       
        layout.addWidget(P5.progress5, 3, 3, 1, 4)
        P5.progress5.show()

        # energy landscape progress:       
        self.label_Hline1 = QtGui.QLabel('')
        self.label_Hline1.setFont(font_standard)
        self.label_Hline1.setMargin(0)
        self.label_Hline1.setFrameStyle(QtGui.QFrame.HLine | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_Hline1, 4, 1, 1, 6, QtCore.Qt.AlignVCenter)
        self.label_Hline1.show()

        P5.button_erg = QtGui.QPushButton('Energy Landscape', self)
        P5.button_erg.clicked.connect(self.start_task6)
        layout.addWidget(P5.button_erg, 5, 1, 1, 2)
        P5.button_erg.setDisabled(True)
        P5.button_erg.show()

        P5.progress6 = QtGui.QProgressBar(minimum=0, maximum=100, value=0)
        self.progress6Changed.connect(self.on_progress6Changed)       
        layout.addWidget(P5.progress6, 5, 3, 1, 4)
        P5.progress6.show()

        # 3d trajectories progress:
        self.label_Hline2 = QtGui.QLabel('')
        self.label_Hline2.setFont(font_standard)
        self.label_Hline2.setMargin(0)
        self.label_Hline2.setFrameStyle(QtGui.QFrame.HLine | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_Hline2, 6, 1, 1, 6, QtCore.Qt.AlignVCenter)
        self.label_Hline2.show()

        P5.button_toP6 = QtGui.QPushButton('View Energy Landscape', self)
        P5.button_toP6.clicked.connect(gotoP6)
        layout.addWidget(P5.button_toP6, 7, 3, 1, 2)
        P5.button_toP6.setDisabled(True)
        P5.button_toP6.show()

        # extend spacing:
        self.label_space = QtGui.QLabel("")
        self.label_space.setFont(font_standard)
        self.label_space.setMargin(0)
        layout.addWidget(self.label_space, 8, 0, 7, 4, QtCore.Qt.AlignVCenter)
        self.label_space.show()

    ##########
    # Task 5:
    @QtCore.Slot()
    def start_task5(self):
        tabs.setTabEnabled(0, False)
        tabs.setTabEnabled(1, False)
        tabs.setTabEnabled(2, False)
        tabs.setTabEnabled(3, False)

        P5.button_CC.setDisabled(True)
        P5.button_CC.setText('Finding Conformational Coordinates')
        P5.entry_temp.setDisabled(True)
        P5.entry_proc.setDisabled(True)
        P5.entry_opt.setDisabled(True)

        set_params.op(0) #send new GUI data to user parameters file

        if P5.user_optFlow == 1 and P4.recompute == 1:
            # hard-remove pre-existing optical flow folders:
            rcDir2 = os.path.join(P1.user_directory, 'outputs_{}/CC/CC_OF'.format(p.proj_name))
            rcDir3 = os.path.join(P1.user_directory, 'outputs_{}/CC/CC_OF_fig'.format(p.proj_name))
            if os.path.isdir(rcDir2):
                shutil.rmtree(rcDir2)
            if os.path.isdir(rcDir3):
                shutil.rmtree(rcDir3)

        task5 = threading.Thread(target=FindConformationalCoord.op, args=(self.progress5Changed, ))
        task5.daemon = True
        task5.start()

    @QtCore.Slot(int)
    def on_progress5Changed(self, val):
        P5.progress5.setValue(val)
        if val == 100:
            p.resProj = 7
            set_params.op(0) #send new GUI data to user parameters file
            time.sleep(5)
            gc.collect()
            P5.button_CC.setText('Conformational Coordinates Complete')
            self.start_task6()
            
    ##########
    # Task 6:
    @QtCore.Slot()
    def start_task6(self):
        tabs.setTabEnabled(0, False)
        tabs.setTabEnabled(1, False)
        tabs.setTabEnabled(2, False)
        tabs.setTabEnabled(3, False)

        P5.button_erg.setDisabled(True)
        P5.button_erg.setText(' Computing Energy Landscape ')
        P5.entry_temp.setDisabled(True)
        P5.entry_proc.setDisabled(True)

        set_params.op(0) #send new GUI data to user parameters file

        task6 = threading.Thread(target=EL1D.op, args=(self.progress6Changed, ))
        ''' ZULU
        if P3.user_dimensions == 1:
            task6 = threading.Thread(target=EL1D.op, args=(self.progress6Changed, ))
        else:
            task6 = threading.Thread(target=EL2D.op, args=(self.progress6Changed, ))
            '''
        task6.daemon = True
        task6.start()

    @QtCore.Slot(int)
    def on_progress6Changed(self, val): #ZULU
        P5.progress6.setValue(val)
        if val == 100:
            p.resProj = 8
            set_params.op(0) #send new GUI data to user parameters file
            time.sleep(5)
            gc.collect()
            P5.button_erg.setText('Energy Landscape Complete')
            
            fnameOM = os.path.join(P1.user_directory,'outputs_{}/ELConc50/OM/S2_OM'.format(p.proj_name)) #occupancy
            fnameEL = os.path.join(P1.user_directory,'outputs_{}/ELConc50/OM/S2_EL'.format(p.proj_name)) #energy
            while not os.path.exists(fnameOM): #wait for file generation
                time.sleep(1)
            P4.Occ1d = np.fromfile(fnameOM, dtype=int)
            P4.Erg1d = np.fromfile(fnameEL)

            Erg1dMain.entry_width.model().item(0).setEnabled(False)
            Erg1dMain.button_traj.setDisabled(False)
            while not os.path.exists(fnameEL): #wait for file generation
                time.sleep(1)
            P4.Erg1d = np.fromfile(fnameEL)
            
            Erg1dMain.plot_erg1d.update_figure() #updates 1d landscape plot
            
            tabs.setTabEnabled(0, True)
            tabs.setTabEnabled(1, True)
            tabs.setTabEnabled(2, True)
            tabs.setTabEnabled(3, True)
            P5.button_toP6.setDisabled(False)

# =============================================================================
# GUI tab 5:
# =============================================================================

class P6(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(P6, self).__init__(parent)
        self.left = 10
        self.top = 10
        self.initUI()

    def initUI(self):
        erg_tab1 = Erg1dMain(self)
        erg_tab2 = Erg2dMain(self)
        global erg_tabs
        erg_tabs = QtGui.QTabWidget(self)
        erg_tabs.addTab(erg_tab1, '1D Energy Path')
        erg_tabs.addTab(erg_tab2, '2D Energy Landscape')
        erg_tabs.setTabEnabled(1, False)
        style = """QTabWidget::tab-bar{
                alignment: center;
                }"""
        self.setStyleSheet(style)
        self.setCentralWidget(erg_tabs)
        self.show()
        
# =============================================================================
# 1D energy landscape figures:
# =============================================================================

class Erg1dMain(QtGui.QDialog):
    occ = False
    react1 = True
    reprepare = 0 #F/T: if PrepareOutputS2 has already been computed (0 if not)
    
    # threading:
    progress7Changed = QtCore.Signal(int)
    
    def __init__(self, parent=None):
        super(Erg1dMain, self).__init__(parent)
        layout = QtGui.QGridLayout(self)
        Erg1dMain.plot_erg1d = Erg1dUpdate(self)
        layout.addWidget(Erg1dMain.plot_erg1d, 1, 0, 8, 8)
        toolbar = NavigationToolbar(Erg1dMain.plot_erg1d, self)
        layout.addWidget(toolbar, 0, 0, 1, 8)

        def choose_CC():
            if Erg1dMain.chooseCC.currentText() == 'CC 1':
                Erg1dMain.react1 = True
                Erg1dMain.plot_erg1d.update_figure()
            elif Erg1dMain.chooseCC.currentText() == 'CC 2':
                Erg1dMain.react1 = False
                Erg1dMain.plot_erg1d.update_figure()

        # switch between energy and occupancy:
        def choose_erg2occ():
            if Erg1dMain.erg2occ.currentText() == 'Energy':
                Erg1dMain.occ = False
                Erg1dMain.plot_erg1d.update_figure()
            elif Erg1dMain.erg2occ.currentText() == 'Occupancy':
                Erg1dMain.occ = True
                Erg1dMain.plot_erg1d.update_figure()

        self.label_edge1 = QtGui.QLabel('')
        self.label_edge1.setMargin(20)
        self.label_edge1.setLineWidth(1)
        self.label_edge1.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge1, 9, 0, 1, 2)
        self.label_edge1.show()

        self.label_rep = QtGui.QLabel('View Conformational Coordinate:')
        self.label_rep.setFont(font_standard)
        self.label_rep.setMargin(20)
        self.label_rep.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_rep, 9, 0, 1, 1)
        self.label_rep.show()

        Erg1dMain.chooseCC = QtGui.QComboBox(self)
        Erg1dMain.chooseCC.setFont(font_standard)
        Erg1dMain.chooseCC.addItem('CC 1')
        Erg1dMain.chooseCC.addItem('CC 2')
        if P3.user_dimensions == 1:
            Erg1dMain.chooseCC.setDisabled(True)
        elif P3.user_dimensions == 2:
            Erg1dMain.chooseCC.setDisabled(False)
        Erg1dMain.chooseCC.setToolTip('Switch between 1D conformational coordinates.')
        Erg1dMain.chooseCC.currentIndexChanged.connect(choose_CC)
        layout.addWidget(Erg1dMain.chooseCC, 9, 1, 1, 1, QtCore.Qt.AlignLeft)
        Erg1dMain.chooseCC.show()

        self.label_edge2 = QtGui.QLabel('')
        self.label_edge2.setMargin(20)
        self.label_edge2.setLineWidth(1)
        self.label_edge2.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge2, 9, 2, 1, 2)
        self.label_edge2.show()

        self.label_distr = QtGui.QLabel('View Distribution:')
        self.label_distr.setFont(font_standard)
        self.label_distr.setMargin(20)
        self.label_distr.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_distr, 9, 2, 1, 1)
        self.label_distr.show()

        Erg1dMain.erg2occ = QtGui.QComboBox(self)
        Erg1dMain.erg2occ.setFont(font_standard)
        Erg1dMain.erg2occ.addItem('Energy')
        Erg1dMain.erg2occ.addItem('Occupancy')
        Erg1dMain.erg2occ.setToolTip('Switch between energy and occupancy representations.')
        Erg1dMain.erg2occ.currentIndexChanged.connect(choose_erg2occ)
        layout.addWidget(Erg1dMain.erg2occ, 9, 3, 1, 1, QtCore.Qt.AlignLeft)
        Erg1dMain.erg2occ.show()

        self.label_edge3 = QtGui.QLabel('')
        self.label_edge3.setMargin(20)
        self.label_edge3.setLineWidth(1)
        self.label_edge3.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge3, 9, 4, 1, 2)
        self.label_edge3.show()

        self.label_width = QtGui.QLabel('Set Path Width:')
        self.label_width.setFont(font_standard)
        self.label_width.setMargin(20)
        self.label_width.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_width, 9, 4, 1, 1)
        self.label_width.show()

        Erg1dMain.entry_width = QtGui.QComboBox(self)
        Erg1dMain.entry_width.setFont(font_standard)
        Erg1dMain.entry_width.addItem('1 State')
        Erg1dMain.entry_width.model().item(0).setEnabled(False)
        Erg1dMain.entry_width.addItem('2 States')
        Erg1dMain.entry_width.addItem('3 States')
        Erg1dMain.entry_width.addItem('4 States')
        Erg1dMain.entry_width.addItem('5 States')
        Erg1dMain.entry_width.setToolTip('Change the range of neighboring states to average for final reconstruction.')
        Erg1dMain.entry_width.currentIndexChanged.connect(self.choose_width)
        layout.addWidget(Erg1dMain.entry_width, 9, 5, 1, 1, QtCore.Qt.AlignLeft)
        Erg1dMain.entry_width.show()

        self.label_edge4 = QtGui.QLabel('')
        self.label_edge4.setMargin(20)
        self.label_edge4.setLineWidth(1)
        self.label_edge4.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge4, 9, 6, 1, 2)
        self.label_edge4.show()

        # 3d trajectories progress:
        Erg1dMain.button_traj = QtGui.QPushButton('Compute 3D Trajectories', self)
        Erg1dMain.button_traj.clicked.connect(self.start_task7)
        layout.addWidget(Erg1dMain.button_traj, 11, 0, 1, 2)
        Erg1dMain.button_traj.setDisabled(False)
        Erg1dMain.button_traj.show()

        Erg1dMain.progress7 = QtGui.QProgressBar(minimum=0, maximum=100, value=0)
        self.progress7Changed.connect(self.on_progress7Changed)
        layout.addWidget(Erg1dMain.progress7, 11, 2, 1, 6)
        Erg1dMain.progress7.show()

    def choose_width(self):
        Erg1dMain.entry_width.model().item(0).setEnabled(True)
        Erg1dMain.entry_width.model().item(1).setEnabled(True)
        Erg1dMain.entry_width.model().item(2).setEnabled(True)
        Erg1dMain.entry_width.model().item(3).setEnabled(True)
        Erg1dMain.entry_width.model().item(4).setEnabled(True)

        if Erg1dMain.entry_width.currentText() == '1 State':
            Erg1dMain.entry_width.model().item(0).setEnabled(False)
            p.width_1D = int(1)
            set_params.op(0)
        if Erg1dMain.entry_width.currentText() == '2 States':
            Erg1dMain.entry_width.model().item(1).setEnabled(False)
            p.width_1D = int(2)
            set_params.op(0)
        if Erg1dMain.entry_width.currentText() == '3 States':
            Erg1dMain.entry_width.model().item(2).setEnabled(False)
            p.width_1D = int(3)
            set_params.op(0)
        if Erg1dMain.entry_width.currentText() == '4 States':
            Erg1dMain.entry_width.model().item(3).setEnabled(False)
            p.width_1D = int(4)
            set_params.op(0)    
        if Erg1dMain.entry_width.currentText() == '5 States':
            Erg1dMain.entry_width.model().item(4).setEnabled(False)
            p.width_1D = int(5)
            set_params.op(0)
            
    ##########
    # Task 7:
    @QtCore.Slot()
    def start_task7(self):
        if Erg1dMain.reprepare == 1: #ZULU
            # overwrite warning:
            msg = 'Final outputs have already been computed for a previous\
                    <i>Path Width</i> selection. To recompute final outputs\
                    with a new path width, previous outputs must be\
                    overwritten.\
                    <br /><br />\
                    Do you want to proceed?'
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Warning' % progname)
            box.setText('<b>Overwrite Warning</b>')
            box.setIcon(QtGui.QMessageBox.Warning)
            box.setFont(font_standard)
            box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
            box.setInformativeText(msg)
            reply = box.exec_()
            if reply == QtGui.QMessageBox.Yes:
                Erg1dMain.progress7.setValue(0)
                # hard-remove pre-existing PrepareOutputS2 outputs:
                prepDir1 = os.path.join(P1.user_directory, 'outputs_{}/bin'.format(p.proj_name))
                if os.path.isdir(prepDir1):
                    shutil.rmtree(prepDir1)
                    os.makedirs(prepDir1)
                    
                Erg1dMain.button_traj.setDisabled(True)
                Erg1dMain.button_traj.setText('Computing 3D Trajectories')
                Erg1dMain.erg2occ.setDisabled(True)
                Erg1dMain.entry_width.setDisabled(True)
                
                tabs.setTabEnabled(0, False)
                tabs.setTabEnabled(1, False)
                tabs.setTabEnabled(2, False)
                tabs.setTabEnabled(3, False)
                tabs.setTabEnabled(4, False)
                
                set_params.op(0) #send new GUI data to parameters file
                time.sleep(5)
        
                task7 = threading.Thread(target=PrepareOutputS2.op, args=(self.progress7Changed, ))
                task7.daemon = True
                task7.start()
                    
            else:
                pass
                    
        else: #if first time running PrepareOutputS2
            Erg1dMain.progress7.setValue(0)
            Erg1dMain.button_traj.setDisabled(True)
            Erg1dMain.button_traj.setText('Computing 3D Trajectories')
            Erg1dMain.erg2occ.setDisabled(True)
            Erg1dMain.entry_width.setDisabled(True)                
            
            tabs.setTabEnabled(0, False)
            tabs.setTabEnabled(1, False)
            tabs.setTabEnabled(2, False)
            tabs.setTabEnabled(3, False)
            tabs.setTabEnabled(4, False)
            
            set_params.op(0) #send new GUI data to parameters file
            time.sleep(5)
    
            task7 = threading.Thread(target=PrepareOutputS2.op, args=(self.progress7Changed, ))
            task7.daemon = True
            task7.start()

    @QtCore.Slot(int)
    def on_progress7Changed(self, val):
        Erg1dMain.progress7.setValue(val)
        if val == 100:
            p.resProj = 9
            set_params.op(0) #send new GUI data to user parameters file
            time.sleep(5)

            Erg1dMain.reprepare = 1
            Erg1dMain.button_traj.setText('Recompute 3D Trajectories')
            Erg1dMain.button_traj.setDisabled(False)
            Erg1dMain.erg2occ.setDisabled(False)
            Erg1dMain.entry_width.setDisabled(False)

            tabs.setTabEnabled(0, True)
            tabs.setTabEnabled(1, True)
            tabs.setTabEnabled(2, True)
            tabs.setTabEnabled(3, True)
            tabs.setTabEnabled(4, True)
            

class Erg1dCanvas(FigureCanvas):
    def __init__(self, parent=None):
        Erg1dCanvas.fig = Figure(dpi=200)
        self.axes = Erg1dCanvas.fig.add_subplot(111)
        self.compute_initial_figure()
        FigureCanvas.__init__(self, Erg1dCanvas.fig)
        self.setParent(parent)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

class Erg1dUpdate(Erg1dCanvas):
    def __init__(self, *args, **kwargs):
        Erg1dCanvas.__init__(self, *args, **kwargs)

    def compute_initial_figure(self):
        Erg1dCanvas.fig.set_tight_layout(True)
        self.axes.set_xlabel('Conformational Coordinate 1', fontsize=6)
        self.axes.set_ylabel('Energy (kcal/mol)', fontsize=6)

    def update_figure(self):                
        if Erg1dMain.occ is True: #plot occupancies
            LS1d = P4.Occ1d
        else: #plot energies
            LS1d = P4.Erg1d #energy path for plot

        self.axes.clear()
        for tick in self.axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        Erg1dCanvas.fig.set_tight_layout(True)
        if Erg1dMain.react1 is True:
            self.axes.set_xlabel('Conformational Coordinate 1', fontsize=6)
        else:
            self.axes.set_xlabel('Conformational Coordinate 2', fontsize=6)
        if Erg1dMain.occ is True:
            im = self.axes.plot(np.arange(1,51), LS1d, linewidth=1, c='#d62728') #C2
            self.axes.set_title('1D Occupancy Map', fontsize=8)
            self.axes.set_ylabel('Occupancy', fontsize=6)
        else:
            im = self.axes.plot(np.arange(1,51), LS1d, linewidth=1, c='#1f77b4') #C0
            self.axes.set_title('1D Energy Path', fontsize=8)
            self.axes.set_ylabel('Energy (kcal/mol)', fontsize=6)

        self.axes.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)
        self.axes.autoscale()
        self.show()
        self.draw()
    
        
class Erg2dMain(QtGui.QDialog):
    occ = False
    reprepare = 0 #F/T: if PrepareOutputS2 has already been computed (0 if not)
    customPath = True
    # threading:
    progress7Changed = QtCore.Signal(int) #ZULU
    
    def __init__(self, parent=None):
        super(Erg2dMain, self).__init__(parent)
        layout = QtGui.QGridLayout(self)
        
        self.label_edge0 = QtGui.QLabel('')
        self.label_edge0.setMargin(20)
        self.label_edge0.setLineWidth(1)
        self.label_edge0.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge0, 0, 0, 10, 7)
        self.label_edge0.show()
        
        Erg2dMain.plot_erg2d = Erg2dUpdate(self)
        layout.addWidget(Erg2dMain.plot_erg2d, 1, 1, 8, 5, QtCore.Qt.AlignLeft)
        toolbar = NavigationToolbar(Erg2dMain.plot_erg2d, self)
        layout.addWidget(toolbar, 9, 1, 1, 5, QtCore.Qt.AlignLeft)

        # switch between energy and occupancy:
        def choose_erg2occ():
            if Erg2dMain.erg2occ.currentText() == 'Energy':
                Erg2dMain.occ = False
                Erg2dMain.plot_erg2d.update_figure()
            elif Erg2dMain.erg2occ.currentText() == 'Occupancy':
                Erg2dMain.occ = True
                Erg2dMain.plot_erg2d.update_figure()

        Erg2dMain.button_reset = QtGui.QPushButton('Reset Points', self)
        Erg2dMain.button_reset.clicked.connect(Erg2dUpdate.reset) #ZULU
        layout.addWidget(Erg2dMain.button_reset, 11, 1, 1, 1)
        Erg2dMain.button_reset.setDisabled(False)
        Erg2dMain.button_reset.show()

        self.label_edge2 = QtGui.QLabel('')
        self.label_edge2.setMargin(20)
        self.label_edge2.setLineWidth(1)
        self.label_edge2.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge2, 11, 2, 1, 2)
        self.label_edge2.show()

        self.label_distr = QtGui.QLabel('View Distribution:')
        self.label_distr.setFont(font_standard)
        self.label_distr.setMargin(20)
        self.label_distr.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_distr, 11, 2, 1, 1)
        self.label_distr.show()

        Erg2dMain.erg2occ = QtGui.QComboBox(self)
        Erg2dMain.erg2occ.setFont(font_standard)
        Erg2dMain.erg2occ.addItem('Energy')
        Erg2dMain.erg2occ.addItem('Occupancy')
        Erg2dMain.erg2occ.setToolTip('Switch between energy and occupancy representations.')
        Erg2dMain.erg2occ.currentIndexChanged.connect(choose_erg2occ)
        layout.addWidget(Erg2dMain.erg2occ, 11, 3, 1, 1, QtCore.Qt.AlignLeft)
        Erg2dMain.erg2occ.show()

        self.label_edge3 = QtGui.QLabel('')
        self.label_edge3.setMargin(20)
        self.label_edge3.setLineWidth(1)
        self.label_edge3.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge3, 11, 4, 1, 2)
        self.label_edge3.show()

        self.label_width = QtGui.QLabel('Set Path Width:')
        self.label_width.setFont(font_standard)
        self.label_width.setMargin(20)
        self.label_width.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_width, 11, 4, 1, 1)
        self.label_width.show()

        Erg2dMain.entry_width = QtGui.QComboBox(self)
        Erg2dMain.entry_width.setFont(font_standard)
        Erg2dMain.entry_width.addItem('1x1 State')
        Erg2dMain.entry_width.model().item(0).setEnabled(False)
        Erg2dMain.entry_width.addItem('2x2 States')
        Erg2dMain.entry_width.addItem('3x3 States')
        Erg2dMain.entry_width.addItem('4x4 States')
        Erg2dMain.entry_width.addItem('5x5 States')
        Erg2dMain.entry_width.setToolTip('Change the range of neighboring states to average for final reconstruction.')
        Erg2dMain.entry_width.currentIndexChanged.connect(self.choose_width)
        layout.addWidget(Erg2dMain.entry_width, 11, 5, 1, 1, QtCore.Qt.AlignLeft)
        Erg2dMain.entry_width.show()
        
        # =====================================================================
        # Custom path analysis:
        # =====================================================================
        self.label_edge4 = QtGui.QLabel('')
        self.label_edge4.setMargin(20)
        self.label_edge4.setLineWidth(1)
        self.label_edge4.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge4, 1, 8, 4, 6)
        self.label_edge4.show()
        
        self.label_edge5 = QtGui.QLabel('')
        self.label_edge5.setMargin(20)
        self.label_edge5.setLineWidth(1)
        self.label_edge5.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge5, 1, 8, 1, 6)
        self.label_edge5.show()
        
        Erg2dMain.btn_cust = QtGui.QCheckBox('   Custom Path Analysis')
        Erg2dMain.btn_cust.clicked.connect(self.setCustom)
        Erg2dMain.btn_cust.clicked.connect(Erg2dUpdate.reset)
        Erg2dMain.btn_cust.setChecked(True)
        Erg2dMain.btn_cust.setDisabled(False)
        layout.addWidget(Erg2dMain.btn_cust, 1, 8, 1, 6, QtCore.Qt.AlignCenter)
        Erg2dMain.btn_cust.show()
        
        self.label_cust2 = QtGui.QLabel('Custom Path Integral:')
        self.label_cust2.setFont(font_standard)
        self.label_cust2.setMargin(20)
        self.label_cust2.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_cust2, 2, 9, 1, 2)
        self.label_cust2.show()
        
        Erg2dMain.entry_cust = QtGui.QDoubleSpinBox(self)
        Erg2dMain.entry_cust.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)
        Erg2dMain.entry_cust.setDecimals(2)
        Erg2dMain.entry_cust.setSuffix(' kcal/Mol')
        Erg2dMain.entry_cust.setDisabled(True)
        Erg2dMain.entry_cust.setStyleSheet("QDoubleSpinBox { width : 150px }")
        layout.addWidget(Erg2dMain.entry_cust, 2, 11, 1, 2, QtCore.Qt.AlignLeft)
        Erg2dMain.entry_cust.show()
        
        Erg2dMain.btn_custSave = QtGui.QPushButton('Save Custom Path', self)
        Erg2dMain.btn_custSave.clicked.connect(self.saveCoords)
        layout.addWidget(Erg2dMain.btn_custSave, 3, 9, 1, 2)
        Erg2dMain.btn_custSave.setDisabled(False)
        Erg2dMain.btn_custSave.show()
        
        Erg2dMain.btn_custComp = QtGui.QPushButton('Compute 3D Trajectories', self)
        Erg2dMain.btn_custComp.clicked.connect(self.custFinalDialog)
        layout.addWidget(Erg2dMain.btn_custComp, 3, 11, 1, 2)
        Erg2dMain.btn_custComp.setDisabled(False)
        Erg2dMain.btn_custComp.show()
        
        # =====================================================================
        # Least action path analysis:
        # =====================================================================
        self.label_edge4b = QtGui.QLabel('')
        self.label_edge4b.setMargin(20)
        self.label_edge4b.setLineWidth(1)
        self.label_edge4b.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge4b, 5, 8, 4, 6)
        self.label_edge4b.show()
        
        self.label_edge5b = QtGui.QLabel('')
        self.label_edge5b.setMargin(20)
        self.label_edge5b.setLineWidth(1)
        self.label_edge5b.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_edge5b, 5, 8, 1, 6)
        self.label_edge5b.show()
        
        Erg2dMain.btn_la = QtGui.QCheckBox('   Least Action Path Analysis')
        Erg2dMain.btn_la.clicked.connect(self.setLA)
        Erg2dMain.btn_la.clicked.connect(Erg2dUpdate.reset)
        Erg2dMain.btn_la.setChecked(False)
        layout.addWidget(Erg2dMain.btn_la, 5, 8, 1, 6, QtCore.Qt.AlignCenter)
        Erg2dMain.btn_la.show()
        
        self.label_la2b = QtGui.QLabel('LA Path Integral:')
        self.label_la2b.setFont(font_standard)
        self.label_la2b.setMargin(20)
        self.label_la2b.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        layout.addWidget(self.label_la2b, 6, 9, 1, 2)
        self.label_la2b.setDisabled(True)
        self.label_la2b.show()
        
        self.entry_la1b = QtGui.QDoubleSpinBox(self)
        self.entry_la1b.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)
        self.entry_la1b.setDecimals(2)
        self.entry_la1b.setSuffix(' kcal/Mol')
        self.entry_la1b.setDisabled(True)
        self.entry_la1b.setStyleSheet("QDoubleSpinBox { width : 150px }")
        layout.addWidget(self.entry_la1b, 6, 11, 1, 2, QtCore.Qt.AlignLeft)
        self.entry_la1b.setDisabled(True)
        self.entry_la1b.show()
        
        Erg2dMain.btn_laImport = QtGui.QPushButton('Import LA Path', self)
        #Erg2dMain.btn_laImport.clicked.connect(self.start_task7) #ZULU
        layout.addWidget(Erg2dMain.btn_laImport, 7, 9, 1, 2)
        Erg2dMain.btn_laImport.setDisabled(True)
        Erg2dMain.btn_laImport.show()
        
        Erg2dMain.btn_leastComp = QtGui.QPushButton('Compute 3D Trajectories', self)
        Erg2dMain.btn_leastComp.clicked.connect(self.laFinalDialog)
        layout.addWidget(Erg2dMain.btn_leastComp, 7, 11, 1, 2)
        Erg2dMain.btn_leastComp.setDisabled(True)
        Erg2dMain.btn_leastComp.show()
        
    # =========================================================================
    # Initiate Progress on 3d Trajectory {Custom; Least Action}:
    # =========================================================================
        self.label_Hline = QtGui.QLabel("") #aesthetic line left
        self.label_Hline.setFont(font_standard)
        self.label_Hline.setMargin(0)
        self.label_Hline.setFrameStyle(QtGui.QFrame.HLine | QtGui.QFrame.Sunken)
        layout.addWidget(self.label_Hline, 9, 9, 1, 4, QtCore.Qt.AlignVCenter)
        self.label_Hline.show()
        
        Erg2dMain.progress7 = QtGui.QProgressBar(minimum=0, maximum=100, value=0)
        #self.progress7Changed.connect(self.on_progress7Changed) #ZULU
        layout.addWidget(Erg2dMain.progress7, 11, 9, 1, 4)
        Erg2dMain.progress7.setDisabled(True) #ZULU undo when progress is initiated
        Erg2dMain.progress7.show()

    def choose_width(self): #ZULU actually compute this for backend?
        Erg2dMain.entry_width.model().item(0).setEnabled(True)
        Erg2dMain.entry_width.model().item(1).setEnabled(True)
        Erg2dMain.entry_width.model().item(2).setEnabled(True)
        Erg2dMain.entry_width.model().item(3).setEnabled(True)
        Erg2dMain.entry_width.model().item(4).setEnabled(True)

        if Erg2dMain.entry_width.currentText() == '1x1 State':
            Erg2dMain.entry_width.model().item(0).setEnabled(False)
            p.width_2D = int(1)
            set_params.op(0)
        if Erg2dMain.entry_width.currentText() == '2x2 States':
            Erg2dMain.entry_width.model().item(1).setEnabled(False)
            p.width_2D = int(2)
            set_params.op(0)
        if Erg2dMain.entry_width.currentText() == '3x3 States':
            Erg2dMain.entry_width.model().item(2).setEnabled(False)
            p.width_2D = int(3)
            set_params.op(0)
        if Erg2dMain.entry_width.currentText() == '4x4 States':
            Erg2dMain.entry_width.model().item(3).setEnabled(False)
            p.width_2D = int(4)
            set_params.op(0)    
        if Erg2dMain.entry_width.currentText() == '5x5 States':
            Erg2dMain.entry_width.model().item(4).setEnabled(False)
            p.width_2D = int(5)
            set_params.op(0)
            
    def setCustom(self): #ZULU need to update progress7 on which one is active
        if self.btn_cust.isChecked():
            Erg2dMain.customPath = True
            Erg2dMain.btn_la.setChecked(False)
            Erg2dMain.btn_leastComp.setDisabled(True)
            Erg2dMain.btn_laImport.setDisabled(True)
            self.label_cust2.setDisabled(False)
            Erg2dMain.entry_cust.setDisabled(True)
            Erg2dMain.entry_width.setDisabled(False)
            Erg2dMain.erg2occ.setDisabled(False)
            Erg2dMain.button_reset.setDisabled(False)
            self.label_la2b.setDisabled(True)
            self.entry_la1b.setDisabled(True)
            Erg2dMain.btn_custSave.setDisabled(False)
            Erg2dMain.btn_custComp.setDisabled(False)
            self.label_width.setDisabled(False)
            self.label_distr.setDisabled(False)
        else:
            Erg2dMain.customPath = False #used to lock out clicking points on plot
            Erg2dMain.btn_la.setChecked(True)
            Erg2dMain.btn_leastComp.setDisabled(False)
            Erg2dMain.btn_laImport.setDisabled(False)
            self.label_cust2.setDisabled(True)
            Erg2dMain.entry_cust.setDisabled(True)
            Erg2dMain.entry_width.setDisabled(True)
            Erg2dMain.erg2occ.setDisabled(True)
            Erg2dMain.button_reset.setDisabled(True)
            self.label_la2b.setDisabled(False)
            self.entry_la1b.setDisabled(True)
            Erg2dMain.btn_custSave.setDisabled(True)
            Erg2dMain.btn_custComp.setDisabled(True)
            self.label_width.setDisabled(True)
            self.label_distr.setDisabled(True)
            
    def setLA(self): #ZULU need to update progress7 on which one is active
        if self.btn_la.isChecked():
            Erg2dMain.customPath = False #used to lock out clicking points on plot
            Erg2dMain.btn_cust.setChecked(False)
            Erg2dMain.btn_leastComp.setDisabled(False)
            Erg2dMain.btn_laImport.setDisabled(False)
            self.label_cust2.setDisabled(True)
            Erg2dMain.entry_cust.setDisabled(True)
            Erg2dMain.entry_width.setDisabled(True)
            Erg2dMain.erg2occ.setDisabled(True)
            Erg2dMain.button_reset.setDisabled(True)
            self.label_la2b.setDisabled(False)
            self.entry_la1b.setDisabled(True)
            Erg2dMain.btn_custSave.setDisabled(True)
            Erg2dMain.btn_custComp.setDisabled(True)
            self.label_width.setDisabled(True)
            self.label_distr.setDisabled(True)
        else:
            Erg2dMain.customPath = True
            Erg2dMain.btn_cust.setChecked(True)
            Erg2dMain.btn_leastComp.setDisabled(True)
            Erg2dMain.btn_laImport.setDisabled(True)
            self.label_cust2.setDisabled(False)
            Erg2dMain.entry_cust.setDisabled(True)
            Erg2dMain.entry_width.setDisabled(False)
            Erg2dMain.erg2occ.setDisabled(False)
            Erg2dMain.button_reset.setDisabled(False)
            self.label_la2b.setDisabled(True)
            self.entry_la1b.setDisabled(True)
            Erg2dMain.btn_custSave.setDisabled(False)
            Erg2dMain.btn_custComp.setDisabled(False)
            self.label_width.setDisabled(False)
            self.label_distr.setDisabled(False)
            
    def saveCoords(self):
        if len(Erg2dUpdate.points_in_line) > 1:
            msg = 'Performing this action will save the coordinates of the current  \
                    handpicked path into the <i>traj</i> folder of your user directory.\
                    <br /><br />\
                    Do you want to proceed?' 
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s' % progname)
            box.setText('<b>Save Custom Path</b>')
            box.setFont(font_standard)
            box.setIcon(QtGui.QMessageBox.Question)
            box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
            box.setInformativeText(msg)
            reply = box.exec_()
            if reply == QtGui.QMessageBox.Yes: 
                timestr = time.strftime("%Y%m%d-%H%M%S")
                custSaveDir = os.path.join(P1.user_directory, 'outputs_%s/traj/custom_path_%s.npy' % (p.proj_name, timestr))
                np.save(custSaveDir, np.asarray(Erg2dUpdate.points_in_line)) #ZULU need to save to p.py, and load back in
            else:
                pass
        else:
            msg = 'At least two points must be selected to complete this request. \
                    <br /><br />\
                    To proceed, first pick a start point and an end point on the \
                    energy landscape.'
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Error' % progname)
            box.setText('<b>Input Error</b>')
            box.setFont(font_standard)
            box.setIcon(QtGui.QMessageBox.Information)
            box.setInformativeText(msg)
            reply = box.exec_()
            
    def custFinalDialog(self):
        if len(Erg2dUpdate.points_in_line) > 1:
            msg = 'Performing this action will generate the 3D structural sequence \
                    for the current handpicked path. \
                    <br /><br />\
                    Do you want to proceed?' 
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s' % progname)
            box.setText('<b>Generate Custom 3D Sequence</b>')
            box.setFont(font_standard)
            box.setIcon(QtGui.QMessageBox.Question)
            box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
            box.setInformativeText(msg)
            reply = box.exec_()
            if reply == QtGui.QMessageBox.Yes:
                print('Under Construction') #ZULU, start backend progress; disable buttons, etc
            else:
                pass
        else:
            msg = 'At least two points must be selected to complete this request. \
                    <br /><br />\
                    To proceed, first pick a start point and an end point on the \
                    energy landscape.'
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Error' % progname)
            box.setText('<b>Input Error</b>')
            box.setFont(font_standard)
            box.setIcon(QtGui.QMessageBox.Information)
            box.setInformativeText(msg)
            reply = box.exec_()

    def laFinalDialog(self): #ZULU -> make so only activated if path of least action present
        #if len(Erg2dUpdate.points_in_line) > 1: #ZULU. equivalent needed (see message above)
        msg = 'Performing this action will generate the 3D structural sequence \
                for the precomputed path of least action.\
                <br /><br />\
                Do you want to proceed?' 
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s' % progname)
        box.setText('<b>Generate LA 3D Sequence</b>')
        box.setFont(font_standard)
        box.setIcon(QtGui.QMessageBox.Question)
        box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
        box.setInformativeText(msg)
        reply = box.exec_()
        if reply == QtGui.QMessageBox.Yes:
            print('Under Construction') #ZULU, start backend progress; disable buttons, etc
        else:
            pass
        '''else:ZULU
            msg = 'A precomputed path of least action must first be imported to complete this request. \
                    <br /><br />\
                    Please read the user manual for more information.'
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Error' % progname)
            box.setText('<b>Input Error</b>')
            box.setFont(font_standard)
            box.setIcon(QtGui.QMessageBox.Information)
            box.setInformativeText(msg)
            reply = box.exec_()'''
            
    ##########
    # Task 7:
    @QtCore.Slot()
    def start_task7(self):
        if Erg2dMain.reprepare == 1: #ZULU
            # overwrite warning:
            msg = 'Final outputs have already been computed for a previous\
                    <i>Path Width</i> selection. To recompute final outputs\
                    with a new path width, previous outputs must be\
                    overwritten.\
                    <br /><br />\
                    Do you want to proceed?'
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Warning' % progname)
            box.setText('<b>Overwrite Warning</b>')
            box.setIcon(QtGui.QMessageBox.Warning)
            box.setFont(font_standard)
            box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
            box.setInformativeText(msg)
            reply = box.exec_()
            if reply == QtGui.QMessageBox.Yes:
                Erg2dMain.progress7.setValue(0)
                # hard-remove pre-existing PrepareOutputS2 outputs:
                prepDir1 = os.path.join(P1.user_directory, 'outputs_{}/bin'.format(p.proj_name)) #ZULU, HSTAU
                if os.path.isdir(prepDir1):
                    shutil.rmtree(prepDir1)
                    os.makedirs(prepDir1)
                    
                Erg2dMain.button_traj.setDisabled(True)
                Erg2dMain.button_traj.setText('Computing 3D Trajectories') #ZULU update for 2 choices
                Erg2dMain.erg2occ.setDisabled(True)
                Erg2dMain.entry_width.setDisabled(True)
                
                tabs.setTabEnabled(0, False)
                tabs.setTabEnabled(1, False)
                tabs.setTabEnabled(2, False)
                tabs.setTabEnabled(3, False)
                tabs.setTabEnabled(4, False)
                
                set_params.op(0) #send new GUI data to parameters file
                time.sleep(5)
        
                task7 = threading.Thread(target=PrepareOutputS2.op, args=(self.progress7Changed, ))
                task7.daemon = True
                task7.start()
                    
            else:
                pass
                    
        else: #if first time running PrepareOutputS2
            Erg2dMain.progress7.setValue(0)
            Erg2dMain.button_traj.setDisabled(True)
            Erg2dMain.button_traj.setText('Computing 3D Trajectories') #ZULU, update for 2 choices
            Erg2dMain.erg2occ.setDisabled(True)
            Erg2dMain.entry_width.setDisabled(True)
            
            tabs.setTabEnabled(0, False)
            tabs.setTabEnabled(1, False)
            tabs.setTabEnabled(2, False)
            tabs.setTabEnabled(3, False)
            tabs.setTabEnabled(4, False)
            
            set_params.op(0) #send new GUI data to parameters file
            time.sleep(5)
    
            task7 = threading.Thread(target=PrepareOutputS2.op, args=(self.progress7Changed, ))
            task7.daemon = True
            task7.start()

    @QtCore.Slot(int)
    def on_progress7Changed(self, val):
        Erg1dMain.progress7.setValue(val)
        if val == 100:
            p.resProj = 9
            set_params.op(0) #send new GUI data to user parameters file
            time.sleep(5)

            Erg2dMain.reprepare = 1
            Erg2dMain.button_traj.setText('Recompute 3D Trajectories') #ZULU update for 2 choices
            Erg2dMain.button_traj.setDisabled(False)
            Erg2dMain.erg2occ.setDisabled(False)
            Erg2dMain.entry_width.setDisabled(False)

            tabs.setTabEnabled(0, True)
            tabs.setTabEnabled(1, True)
            tabs.setTabEnabled(2, True)
            tabs.setTabEnabled(3, True)
            tabs.setTabEnabled(4, True)        
             
# =============================================================================
# 2D energy landscape figures:   
# =============================================================================
        
class Erg2dCanvas(FigureCanvas): 
    def __init__(self, parent=None):
        Erg2dCanvas.fig = Figure(dpi=200)
        self.axes = Erg2dCanvas.fig.add_subplot(111)
        self.compute_initial_figure()
        FigureCanvas.__init__(self, Erg2dCanvas.fig)
        self.setParent(parent)

    def compute_initial_figure(self):
        pass
    
    
class Erg2dUpdate(Erg2dCanvas):
    coordsX = []
    coordsY = []
    points_in_line = []
    
    def __init__(self, *args, **kwargs):
        Erg2dCanvas.__init__(self, *args, **kwargs)

    def compute_initial_figure(self):
        LS2d = np.random.uniform(0, 2, (70, 70)) #temporary figure to initiate
        im = self.axes.imshow(LS2d, origin='lower', interpolation='nearest', cmap='jet')
        self.cbar = Erg2dCanvas.fig.colorbar(im)
        self.axes.set_xlabel('CC 1', fontsize=6)
        self.axes.set_ylabel('CC 2', fontsize=6)
        Erg2dCanvas.fig.subplots_adjust(right=0.8)

    def update_figure(self):    
        self.axes.clear()
        self.cbar.remove()
            
        if Erg2dMain.occ is True: #plot occupancies
            #LS1d = P4.Occ1d
            LS2d = np.random.uniform(0, 2, (70, 70)) #ZULU, P4.Occ2d
        else: #plot energies
            #ZULU #P4.Erg2d #energy path for plot
            fnameLS = os.path.join(pyDir,'demo/EL_Ribo_70x70.txt')
            try:
                LS2d = loadtxt(fnameLS,float,delimiter=',')
            except ValueError:
                LS2d = loadtxt(fnameLS,float) 
                
            Erg2dUpdate.LS2d_erg = LS2d #ZULU change

        for tick in self.axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(4)

        if Erg2dMain.occ is True:
            im = self.axes.imshow(LS2d, origin='lower', interpolation='nearest', cmap='jet', aspect='equal')
            Erg2dCanvas.fig.subplots_adjust(right=0.725)
            cbar_ax = Erg2dCanvas.fig.add_axes([0.80, 0.15, 0.05, 0.7])
            self.cbar = Erg2dCanvas.fig.colorbar(im, cax=cbar_ax)
            self.axes.set_title('2D Occupancy Map', fontsize=7)
            self.cbar.ax.tick_params(labelsize=4)
            self.cbar.ax.set_title(label='Occupancy',size=4)
        else:
            im = self.axes.imshow(LS2d, origin='lower', interpolation='nearest', cmap='jet', aspect='equal')
            Erg2dCanvas.fig.subplots_adjust(right=0.725)
            cbar_ax = Erg2dCanvas.fig.add_axes([0.80, 0.15, 0.05, 0.7])
            self.cbar = Erg2dCanvas.fig.colorbar(im, cax=cbar_ax)
            self.axes.set_title('2D Energy Landscape', fontsize=7)
            self.cbar.ax.tick_params(labelsize=4)
            self.cbar.ax.set_title(label='Energy\n (kcal/Mol)',size=4)
            
        self.axes.set_xlabel('CC 1', fontsize=6, labelpad=0)
        self.axes.set_ylabel('CC 2', fontsize=6, labelpad=0)
        self.axes.set_xticks([0, np.shape(LS2d)[0]-1])
        self.axes.set_yticks([0, np.shape(LS2d)[1]-1])
        
        # plot new markers and lines via mouse clicks:
        self.axes.plot(Erg2dUpdate.coordsX, Erg2dUpdate.coordsY, color='k', linewidth=.5, zorder=1)
        self.axes.scatter(Erg2dUpdate.coordsX, Erg2dUpdate.coordsY, marker='s', c='w', s=1, zorder=2)
        
        self.show()
        self.draw()
        self.mpl_connect('button_press_event', self.onclick)
        
    def onclick(self, event):
        if Erg2dMain.customPath is True: #lock out clicks on plot if LA Path selected (if False)
            ix, iy = event.xdata, event.ydata
            if ix != None and iy != None:
                self.coordsX.append(np.rint(ix))
                self.coordsY.append(np.rint(iy))
                
                if len(self.coordsX) > 1:
                    bresenham = self.ptsLine(self.coordsX[-2],self.coordsY[-2],self.coordsX[-1],self.coordsY[-1])
                    if len(self.coordsX) == 2:
                        for b1, b2 in bresenham:
                            self.points_in_line.append([b1, b2])
                    if len(self.coordsX) > 2: #don't overcount
                        idx = 0
                        for b1, b2 in bresenham:
                            if idx > 0:
                                self.points_in_line.append([b1, b2])
                            idx += 1
                            
                    erg = 0
                    for e1, e2 in self.points_in_line: #integrate energy of current path
                        erg += Erg2dUpdate.LS2d_erg[int(e2),int(e1)]
                        #print(e1, e2, Erg2dUpdate.LS2d_erg[int(e2),int(e1)]) #ZULU double check on real data
                    Erg2dMain.entry_cust.setValue(erg) #update energetics
                    Erg2dMain.plot_erg2d.update_figure()
                    
                else: #first point
                    erg = Erg2dUpdate.LS2d_erg[int(np.rint(iy)),int(np.rint(ix))]
                    Erg2dMain.entry_cust.setValue(float(erg))
                    Erg2dMain.plot_erg2d.update_figure()
        
    def reset(self):
        Erg2dUpdate.coordsX = []
        Erg2dUpdate.coordsY = []
        Erg2dUpdate.points_in_line = []
        Erg2dMain.entry_cust.setValue(0.0)
        Erg2dMain.plot_erg2d.update_figure()
        
    def ptsLine(self, x0, y0, x1, y1):
        "Bresenham's line algorithm"
        points_in_line = []
        #energy_in_line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points_in_line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points_in_line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points_in_line.append((x, y))
        return points_in_line
    

        '''
        global completeDialog 
        def completeDialog():
            msg = '3D structural sequence exported successfully. Please refer to the\
            <i>Post-processing</i> section of the user manual for further guidance.'
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s' % progname)
            box.setText('<b>Congratulations!</b>')
            box.setFont(font_standard)
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setInformativeText(msg)
            reply = box.exec_()'''
            
# =============================================================================
# Polar coordinates plot for anchors and deletions:
# =============================================================================

class PDSeleMain(QtGui.QMainWindow):
    def __init__(self):
        super(PDSeleMain, self).__init__()
        self.left = 10
        self.top = 10
        self.initUI()

        # Sub-Help Menu:
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction('&PD Selections', self.guide_PDSele)

    def guide_PDSele(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>PD Selections</b>')
        box.setFont(font_standard)
        box.setInformativeText("<span style='font-weight:normal;'>\
                                On the <i>PD Editor</i> tab, all PD selections can be reviewed in list form, reset, saved, or loaded via the\
                                corresponding buttons.\
                                <br /><br />\
                                On the <i>PD Viewer</i> tab, the polar coordinates plot shows the location of each user-defined PD classification: \
                                <i>To Be Determined</i> (TBD), <i>Anchors</i>, and <i>Removals</i>.\
                                </span>")
        box.setStandardButtons(QtGui.QMessageBox.Ok)
        ret = box.exec_()

    def initUI(self):
        sele_tab1 = PDSeleCanvas1(self)
        sele_tab2 = PDSeleCanvas2(self)
        global sele_tabs
        sele_tabs = QtGui.QTabWidget(self)
        sele_tabs.addTab(sele_tab1, 'PD Editor')
        sele_tabs.addTab(sele_tab2, 'PD Viewer')

        style = """QTabWidget::tab-bar{
                alignment: center;
                }"""
        self.setStyleSheet(style)
        self.setCentralWidget(sele_tabs)
        #self.setWindowModality(QtCore.Qt.ApplicationModal) #freezes out parent window
        self.show()

        sele_tabs.currentChanged.connect(self.onTabChange) #signal for tab changed via direct click

    def onTabChange(self, i):
        if i == 1: #signals when view switched to tab 2
            # re-visualize selections:
            PrD_map = os.path.join(P1.user_directory,'outputs_{}/topos/Euler_PrD/PrD_map.txt'.format(p.proj_name))
            PrD_map_eul = []
            with open(PrD_map) as values:
                for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                    PrD_map_eul.append(column)

            phi_all0 = PrD_map_eul[2]
            theta_all0 = PrD_map_eul[1]
            c_all0 = PrD_map_eul[7]

            phi_all = []
            theta_all = []
            c_all = []
            phi_anch = []
            theta_anch = []
            c_anch = []
            phi_trash = []
            theta_trash = []
            c_trash = []
            
            idx = 1
            for i in phi_all0: #subtraction by 180 needed for scatter's labels
                if idx in P1.a3:
                    phi_anch.append((float(i)-180.)*np.pi/180.)
                elif idx in P1.a4:
                    phi_trash.append((float(i)-180.)*np.pi/180.)
                else:
                    phi_all.append((float(i)-180.)*np.pi/180.) #needed in Radians
                idx+=1
                    
            idx = 1
            for i in theta_all0:
                if idx in P1.a3:
                    theta_anch.append(float(i))
                elif idx in P1.a4:
                    theta_trash.append(float(i))
                else:
                    theta_all.append(float(i))
                idx+=1

            idx = 1
            for i in c_all0:
                if idx in P1.a3:
                    c_anch.append(float(i))
                elif idx in P1.a4:
                    c_trash.append(float(i))
                else:
                    c_all.append(float(i))
                idx+=1

            def format_coord(x,y): #calibrates toolbar coordinates
                return 'Phi={:1.2f}, Theta={:1.2f}'.format(((x*180)/np.pi)-180,y)
                    
            # replot PDSeleCanvas:
            PDSeleCanvas2.axes.clear()
            PDSeleCanvas2.axes.format_coord = format_coord
            # thetas = [0,45,90,135,180,225,270,315] #in same order as labels below (ref only)
            theta_labels = ['%s180%s'%(u"\u00B1",u"\u00b0"),'-135%s'%(u"\u00b0"),'-90%s'%(u"\u00b0"),'-45%s'%(u"\u00b0"),
                            '0%s'%(u"\u00b0"),'45%s'%(u"\u00b0"),'90%s'%(u"\u00b0"),'135%s'%(u"\u00b0")]
            PDSeleCanvas2.axes.set_ylim(0,180)
            PDSeleCanvas2.axes.set_yticks(np.arange(0,180,20))
            PDSeleCanvas2.axes.set_xticklabels(theta_labels)
            for tick in PDSeleCanvas2.axes.xaxis.get_major_ticks():
                tick.label.set_fontsize(4)
            for tick in PDSeleCanvas2.axes.yaxis.get_major_ticks():
                tick.label.set_fontsize(4)
            PDSeleCanvas2.axes.grid(alpha=.2)
            PDSeleCanvas2.axes.tick_params(pad=.3) #distance of theta ticks from circle's edge

            PDSeleAll = PDSeleCanvas2.axes.scatter(phi_all, theta_all, edgecolor='k',\
                                                   linewidth=.1, c=c_all, s=5, label='TBD')
            PDSeleAll.set_alpha(0.75)
            PDSeleAnch = PDSeleCanvas2.axes.scatter(phi_anch, theta_anch, edgecolor='k',\
                                                    linewidth=.3, c='lightgray', s=5, marker='D', label='Anchor')
            PDSeleAnch.set_alpha(0.75)
            PDSeleTrash = PDSeleCanvas2.axes.scatter(phi_trash, theta_trash, edgecolor='k',\
                                                    linewidth=.5, c='k', s=5, marker='x', label='Removal') #x or X
            PDSeleTrash.set_alpha(1.)

            PDSeleCanvas2.axes.legend(loc='best', prop={'size': 4})
            #PDSeleCanvas2.figure.set_tight_layout(True)
            PDSeleCanvas2.canvas.draw()


class PDSeleCanvas1(QtGui.QDialog):
    def __init__(self, parent=None):
        super(PDSeleCanvas1, self).__init__(parent)
        self.left = 10
        self.top = 10

        PDSeleCanvas1.progBar1 = QtGui.QProgressBar(self) #minimum=0,maximum=1,value=0)
        PDSeleCanvas1.progBar1.setRange(0,100)
        PDSeleCanvas1.progBar1.setVisible(False)
        PDSeleCanvas1.progBar1.setValue(0)

        PDSeleCanvas1.progBar2 = QtGui.QProgressBar(self) #minimum=0,maximum=1,value=0)
        PDSeleCanvas1.progBar2.setRange(0,100)
        PDSeleCanvas1.progBar2.setVisible(False)
        PDSeleCanvas1.progBar2.setValue(0)

        label_edgeLarge1 = QtGui.QLabel('')
        label_edgeLarge1.setMargin(20)
        label_edgeLarge1.setLineWidth(1)
        label_edgeLarge1.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)

        label_anch = QtGui.QLabel('PD Anchors:')
        label_anch.setFont(font_standard)
        label_anch.setMargin(20)
        label_anch.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        label_anch.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        label_edgeLarge2 = QtGui.QLabel('')
        label_edgeLarge2.setMargin(20)
        label_edgeLarge2.setLineWidth(1)
        label_edgeLarge2.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)

        label_trash = QtGui.QLabel('PD Removals:')
        label_trash.setFont(font_standard)
        label_trash.setMargin(20)
        label_trash.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        label_trash.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        label_edgeLarge3 = QtGui.QLabel('')
        label_edgeLarge3.setMargin(20)
        label_edgeLarge3.setLineWidth(1)
        label_edgeLarge3.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)

        label_occ = QtGui.QLabel('PD Occupancies:')
        label_occ.setFont(font_standard)
        label_occ.setMargin(20)
        label_occ.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        label_occ.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        label_edgeLarge4 = QtGui.QLabel('')
        label_edgeLarge4.setMargin(20)
        label_edgeLarge4.setLineWidth(1)
        label_edgeLarge4.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)

        label_rebed = QtGui.QLabel('PD Embeddings:')
        label_rebed.setFont(font_standard)
        label_rebed.setMargin(20)
        label_rebed.setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Sunken)
        label_rebed.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.btn_anchList = QtGui.QPushButton('List Anchors')
        self.btn_anchList.clicked.connect(self.anchorList)
        self.btn_anchReset = QtGui.QPushButton('Reset Anchors')
        self.btn_anchReset.clicked.connect(self.anchorReset)
        self.btn_anchSave = QtGui.QPushButton('Save Anchors')
        self.btn_anchSave.clicked.connect(self.anchorSave)
        self.btn_anchLoad = QtGui.QPushButton('Load Anchors')
        self.btn_anchLoad.clicked.connect(self.anchorLoad)

        self.btn_trashList = QtGui.QPushButton('List Removals')
        self.btn_trashList.clicked.connect(self.trashList)
        self.btn_trashReset = QtGui.QPushButton('Reset Removals')
        self.btn_trashReset.clicked.connect(self.trashReset)
        self.btn_trashSave = QtGui.QPushButton('Save Removals')
        self.btn_trashSave.clicked.connect(self.trashSave)
        self.btn_trashLoad = QtGui.QPushButton('Load Removals')
        self.btn_trashLoad.clicked.connect(self.trashLoad)

        self.btn_occList = QtGui.QPushButton('List Occupancies')
        self.btn_occList.clicked.connect(self.occListGen)
        self.btn_rebedList = QtGui.QPushButton('List Re-embeddings')
        self.btn_rebedList.clicked.connect(self.rebedListGen)
        
        # forced space bottom:
        label_spaceBtm = QtGui.QLabel("")
        label_spaceBtm.setFont(font_standard)
        label_spaceBtm.setMargin(0)
        label_spaceBtm.show()

        layout = QtGui.QGridLayout()
        layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)

        layout.addWidget(label_edgeLarge1, 0, 0, 1, 6, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_anch, 0, 0, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_anchList, 0, 1, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_anchReset, 0, 2, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_anchSave, 0, 3, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_anchLoad, 0, 4, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(PDSeleCanvas1.progBar1, 1, 0, 1, 6)

        layout.addWidget(label_edgeLarge2, 2, 0, 1, 6, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_trash, 2, 0, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_trashList, 2, 1, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_trashReset, 2, 2, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_trashSave, 2, 3, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_trashLoad, 2, 4, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(PDSeleCanvas1.progBar2, 3, 0, 1, 6)

        layout.addWidget(label_edgeLarge3, 4, 0, 1, 3, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_occ, 4, 0, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_occList, 4, 1, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_edgeLarge4, 4, 3, 1, 3, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_rebed, 4, 3, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.btn_rebedList, 4, 4, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_spaceBtm, 5, 1, 3, 6, QtCore.Qt.AlignVCenter)

        self.setLayout(layout)

    def anchorList(self):
        PrDs = []
        CC1s = []
        S1s = []
        CC2s = []
        S2s = []
        colors = []
        P4.anch_list = []
        
        idx_sum = 0
        for i in range(1,P3.PrD_total+1):
            if P4.anchorsAll[i].isChecked():
                idx_sum += 1

        if idx_sum == 0:
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Error' % progname)
            box.setText('<b>Input Error</b>')
            box.setFont(font_standard)
            box.setIcon(QtGui.QMessageBox.Information)
            box.setInformativeText('No PD anchors have been selected.\
                                    <br /><br />\
                                    Select anchors using the <i>Set PD Anchors</i> box\
                                    on the left side of the <i>Eigenvectors</i> tab.')
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setDefaultButton(QtGui.QMessageBox.Ok)
            ret = box.exec_()
                
        else:
            idx = 0
            for i in range(1,P3.PrD_total+1):
                if P4.anchorsAll[i].isChecked():
                    PrDs.append(int(i))
                    # CC1s:                        
                    CC1s.append(int(P4.reactCoord1All[i].value()))
                    # S1s:
                    if P4.senses1All[i].currentText() == 'S1: FWD':
                        S1s.append(int(1))
                    else:
                        S1s.append(int(-1))
                    # CC2s:
                    CC2s.append(int(P4.reactCoord2All[i].value()))
                    # S2s:
                    if P4.senses2All[i].currentText() == 'S2: FWD':
                        S2s.append(int(1))
                    else:
                        S2s.append(int(-1))
                    # colors:
                    colors.append(P3.col[int(i-1)])
                    idx += 1

            if P3.user_dimensions == 1:
                P4.anch_list = zip(PrDs,CC1s,S1s,colors)
            elif P3.user_dimensions == 2:
                P4.anch_list = zip(PrDs,CC1s,S1s,CC2s,S2s,colors)

            self.table = anchTable(data = P4.anch_list)
            sizeObject = QtGui.QDesktopWidget().screenGeometry(-1) #user screen size
            self.table.move((sizeObject.width()/2)-100,(sizeObject.height()/2)-300)
            self.table.show()

    # reset assignments of all PD anchors:
    def anchorReset(self):
        box = QtGui.QMessageBox(self)
        self.setWindowTitle('Reset PD Anchors')
        box.setText('<b>Reset Warning</b>')
        box.setFont(font_standard)
        box.setIcon(QtGui.QMessageBox.Warning)
        msg = 'Performing this action will deselect all active anchors.\
                The user-defined values set within each (if any) will still remain.\
                <br /><br />\
                Do you want to proceed?'
        box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
        box.setInformativeText(msg)
        reply = box.exec_()

        if reply == QtGui.QMessageBox.Yes:
            for i in range(1,P3.PrD_total+1):
                if P4.anchorsAll[i].isChecked():
                    P4.anchorsAll[i].setChecked(False)
                    P4.reactCoord1All[i].setDisabled(False)
                    P4.senses1All[i].setDisabled(False)
                    if P3.user_dimensions == 2:
                        P4.reactCoord2All[i].setDisabled(False)
                        P4.senses2All[i].setDisabled(False)

            PDSeleCanvas1.progBar1.setVisible(False)
            PDSeleCanvas1.progBar1.setValue(0)

            P1.x3 = []
            P1.y3 = []
            P1.z3 = []
            P1.a3 = []
            P4.viz2.update_scene3()

        elif reply == QtGui.QMessageBox.No:
            pass

    def anchorSave(self):
        temp_anch_list = []
        anch_sum = 0
        for i in range(1,P3.PrD_total+1):
            if P4.anchorsAll[i].isChecked():
                anch_sum += 1
        if anch_sum == 0:
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Warning' % progname)
            box.setIcon(QtGui.QMessageBox.Warning)
            box.setText('<b>Input Warning</b>')
            box.setFont(font_standard)
            msg = 'At least one anchor must first be selected before saving.'
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setInformativeText(msg)
            reply = box.exec_()
        elif anch_sum > 0:
            box = QtGui.QMessageBox(self)
            self.setWindowTitle('%s Save Data' % progname)
            box.setText('<b>Save Current Anchors</b>')
            box.setFont(font_standard)
            box.setIcon(QtGui.QMessageBox.Information)
            msg = 'Performing this action will save a list of all active anchors\
                    to the <i>outputs/CC</i> directory for future reference.\
                    <br /><br />\
                    Do you want to proceed?'
            box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
            box.setInformativeText(msg)
            reply = box.exec_()

            if reply == QtGui.QMessageBox.Yes:

                PrDs = []
                CC1s = []
                S1s = []
                CC2s = []
                S2s = []
                colors = []
                P4.anch_list = []

                idx = 0
                for i in range(1,P3.PrD_total+1):
                    if P4.anchorsAll[i].isChecked():
                        PrDs.append(int(i))
                        # CC1s:
                        CC1s.append(int(P4.reactCoord1All[i].value()))
                        # S1s:
                        if P4.senses1All[i].currentText() == 'S1: FWD':
                            S1s.append(int(1))
                        else:
                            S1s.append(int(-1))
                        # CC2s:
                        CC2s.append(int(P4.reactCoord2All[i].value()))
                        # S2s:
                        if P4.senses2All[i].currentText() == 'S2: FWD':
                            S2s.append(int(1))
                        else:
                            S2s.append(int(-1))
                        # colors:
                        colors.append(P3.col[int(i-1)])
                        idx += 1

                if P3.user_dimensions == 1:
                    temp_anch_list = zip(PrDs,CC1s,S1s,colors)
                elif P3.user_dimensions == 2:
                    temp_anch_list = zip(PrDs,CC1s,S1s,CC2s,S2s,colors)

                timestr = time.strftime("%Y%m%d-%H%M%S")
                tempAnchInputs = os.path.join(P1.user_directory, 'outputs_%s/CC/temp_anchors_%s.txt' % (p.proj_name,timestr))
                
                np.savetxt(tempAnchInputs, list(temp_anch_list), fmt='%i', delimiter='\t')

                box = QtGui.QMessageBox(self)
                box.setWindowTitle('%s Save Current Anchors' % progname)
                box.setIcon(QtGui.QMessageBox.Information)
                box.setText('<b>Saving Complete</b>')
                box.setFont(font_standard)
                msg = 'Current anchor selections have been saved to the <i>outputs/CC</i> directory.'
                box.setStandardButtons(QtGui.QMessageBox.Ok)
                box.setInformativeText(msg)
                reply = box.exec_()

            elif reply == QtGui.QMessageBox.No:
                pass

    def anchorLoad(self):
        self.btn_anchList.setDisabled(True)
        self.btn_anchReset.setDisabled(True)
        self.btn_anchSave.setDisabled(True)
        self.btn_anchLoad.setDisabled(True)
        self.btn_trashList.setDisabled(True)
        self.btn_trashReset.setDisabled(True)
        self.btn_trashSave.setDisabled(True)
        self.btn_trashLoad.setDisabled(True)
        self.btn_occList.setDisabled(True)
        self.btn_rebedList.setDisabled(True)

        anch_sum = 0
        for i in range(1,P3.PrD_total+1):
            if P4.anchorsAll[i].isChecked():
                anch_sum += 1
        if anch_sum == 0:
            PDSeleCanvas1.fname = QtGui.QFileDialog.getOpenFileName(self, 'Choose Data File', '',
                                     ('Data Files (*.txt)'))[0]
            if PDSeleCanvas1.fname:
                try:
                    if P3.user_dimensions == 1:
                        data = []
                        with open(PDSeleCanvas1.fname) as values:
                            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                                data.append(column)
                        PrDs = data[0]
                        CC1s = data[1]
                        S1s = data[2]

                        data_all = np.column_stack((PrDs,CC1s,S1s))
                        PrD = []
                        CC1 = []
                        S1 = []
                        Color = []
                        idx = 0
                        for i,j,k in data_all:
                            PrD.append(int(i))
                            CC1.append(int(j))
                            S1.append(int(k))
                            idx += 1

                        P4.anch_list = zip(PrD,CC1,S1)
                        P4.anch_list, anch_zip = itertools.tee(P4.anch_list)
                        p.anch_list = list(anch_zip) #PrD,CC1,S1 for 1D; PrD,CC1,S1,CC2,S2 for 2D

                        idx = 0
                        prog = 0
                        PDSeleCanvas1.progBar1.setValue(prog)
                        PDSeleCanvas1.progBar1.setVisible(True)

                        for i in PrD:
                            P4.entry_PrD.setValue(int(i))
                            P4.user_PrD = i
                            P4.PrD_hist = i
                            if P4.trashAll[i].isChecked() == False: #avoids conflict
                                P4.anchorsAll[i].setChecked(True)
                            P4.reactCoord1All[i].setValue(CC1[idx])
                            if S1[idx] == 1:
                                P4.senses1All[i].setCurrentIndex(0)
                            elif S1[idx] == -1:
                                P4.senses1All[i].setCurrentIndex(1)
                            prog += (1./len(PrD))*100
                            PDSeleCanvas1.progBar1.setValue(prog)
                            idx += 1

                        P4.entry_PrD.setValue(1)

                    elif P3.user_dimensions == 2:
                        fname = os.path.join(P1.user_directory,'outputs_{}/CC/user_anchors.txt'.format(p.proj_name))
                        data = []
                        with open(fname) as values:
                            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                                data.append(column)
                        PrDs = data[0]
                        CC1s = data[1]
                        S1s = data[2]
                        CC2s = data[3]
                        S2s = data[4]

                        data_all = np.column_stack((PrDs,CC1s,S1s,CC2s,S2s))
                        PrD = []
                        CC1 = []
                        S1 = []
                        CC2 = []
                        S2 = []
                        idx = 0
                        for i,j,k,l,m in data_all:
                            PrD.append(int(i))
                            CC1.append(int(j))
                            S1.append(int(k))
                            CC2.append(int(l))
                            S2.append(int(m))

                        P4.anch_list = zip(PrD,CC1,S1,CC2,S2)
                        P4.anch_list, anch_zip = itertools.tee(P4.anch_list)
                        p.anch_list = list(anch_zip) #PrD,CC1,S1 for 1D; PrD,CC1,S1,CC2,S2 for 2D

                        idx = 0
                        prog = 0
                        PDSeleCanvas1.progBar1.setValue(prog)
                        PDSeleCanvas1.progBar1.setVisible(True)
                        for i in PrD:
                            P4.user_PrD = i
                            P4.PrD_hist = i
                            if P4.trashAll[i].isChecked() == False: #avoids conflict
                                P4.anchorsAll[i].setChecked(True)
                            P4.reactCoord1All[i].setValue(CC1[idx])
                            P4.reactCoord2All[i].setValue(CC2[idx])
                            if S1[idx] == 1:
                                P4.senses1All[i].setCurrentIndex(0)
                            elif S1[idx] == -1:
                                P4.senses1All[i].setCurrentIndex(1)
                            if S2[idx] == 1:
                                P4.senses2All[i].setCurrentIndex(0)
                            elif S2[idx] == -1:
                                P4.senses2All[i].setCurrentIndex(1)
                            prog += (1./len(PrD))*100
                            PDSeleCanvas1.progBar1.setValue(prog)
                            idx += 1
                        P4.entry_PrD.setValue(1)

                    PDSeleCanvas1.progBar1.setValue(100)

                    box = QtGui.QMessageBox(self)
                    box.setWindowTitle('%s Load Previous Anchors' % progname)
                    box.setIcon(QtGui.QMessageBox.Information)
                    box.setText('<b>Loading Complete</b>')
                    box.setFont(font_standard)
                    msg = 'Previous anchor selections have been loaded on the <i>Eigenvectors</i> tab.'
                    box.setStandardButtons(QtGui.QMessageBox.Ok)
                    box.setInformativeText(msg)
                    reply = box.exec_()

                except:
                    box = QtGui.QMessageBox(self)
                    box.setWindowTitle('%s Error' % progname)
                    box.setText('<b>Input Error</b>')
                    box.setIcon(QtGui.QMessageBox.Warning)
                    box.setFont(font_standard)
                    box.setInformativeText('Incorrect file structure detected.')
                    box.setStandardButtons(QtGui.QMessageBox.Ok)
                    box.setDefaultButton(QtGui.QMessageBox.Ok)
                    ret = box.exec_()

                    PDSeleCanvas1.progBar1.setVisible(False)
                    PDSeleCanvas1.progBar1.setValue(0)
            else:
                pass

        elif anch_sum > 0:
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Warning' % progname)
            box.setIcon(QtGui.QMessageBox.Warning)
            box.setText('<b>Input Warning</b>')
            box.setFont(font_standard)
            msg = 'To load anchors from a previous session, first clear all currently selected\
                    anchors via the <i>Reset Anchors</i> button.'
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setInformativeText(msg)
            reply = box.exec_()

        self.btn_anchList.setDisabled(False)
        self.btn_anchReset.setDisabled(False)
        self.btn_anchSave.setDisabled(False)
        self.btn_anchLoad.setDisabled(False)
        self.btn_trashList.setDisabled(False)
        self.btn_trashReset.setDisabled(False)
        self.btn_trashSave.setDisabled(False)
        self.btn_trashLoad.setDisabled(False)
        self.btn_occList.setDisabled(False)
        self.btn_rebedList.setDisabled(False)

    def trashList(self):
        PrDs = []
        trashed = []
        
        trash_sum = 0
        for i in range(1,P3.PrD_total+1):
            if P4.trashAll[i].isChecked():
                trash_sum += 1

        if trash_sum == 0:
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Error' % progname)
            box.setText('<b>Input Error</b>')
            box.setFont(font_standard)
            box.setIcon(QtGui.QMessageBox.Information)
            box.setInformativeText('No PDs have been selected for removal.\
                                    <br /><br />\
                                    Select PDs for removal using the <i>Remove PD</i> option\
                                    on the left side of the <i>Eigenvectors</i> tab.')
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setDefaultButton(QtGui.QMessageBox.Ok)
            ret = box.exec_()
                
        else:
            for i in range(1,P3.PrD_total+1):
                if P4.trashAll[i].isChecked():
                    trashed.append('True')
                else:
                    trashed.append('False')
                PrDs.append(int(i))

            sorted_trash = sorted(zip(PrDs,trashed), key=lambda x: x[1], reverse=True)
            self.table = trashTable(data = sorted_trash)
            sizeObject = QtGui.QDesktopWidget().screenGeometry(-1) #user screen size
            self.table.move((sizeObject.width()/2)-100,(sizeObject.height()/2)-300)
            self.table.show()

    # reset assignments of all PD removals:
    def trashReset(self):
        box = QtGui.QMessageBox(self)
        self.setWindowTitle('Reset PD Removals')
        box.setText('<b>Reset Warning</b>')
        box.setFont(font_standard)
        box.setIcon(QtGui.QMessageBox.Warning)
        msg = 'Performing this action will deselect all active removals.\
                <br /><br />\
                Do you want to proceed?'
        box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
        box.setInformativeText(msg)
        reply = box.exec_()

        if reply == QtGui.QMessageBox.Yes:
            for i in range(1,P3.PrD_total+1):
                if P4.trashAll[i].isChecked():
                    P4.trashAll[i].setChecked(False)
                    P4.anchorsAll[i].setDisabled(False)

            PDSeleCanvas1.progBar2.setVisible(False)
            PDSeleCanvas1.progBar2.setValue(0)

            P1.x4 = []
            P1.y4 = []
            P1.z4 = []
            P1.a4 = []
            P4.viz2.update_scene3()

        elif reply == QtGui.QMessageBox.No:
            pass
        
    def trashSave(self):
        trash_sum = 0
        for i in range(1,P3.PrD_total+1):
            if P4.trashAll[i].isChecked():
                trash_sum += 1
        if trash_sum == 0:
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Warning' % progname)
            box.setIcon(QtGui.QMessageBox.Warning)
            box.setText('<b>Input Warning</b>')
            box.setFont(font_standard)
            msg = 'At least one PD must first be selected for removal before saving.'
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setInformativeText(msg)
            reply = box.exec_()
        elif trash_sum > 0:
            box = QtGui.QMessageBox(self)
            self.setWindowTitle('%s Save Data' % progname)
            box.setText('<b>Save Current Removals</b>')
            box.setFont(font_standard)
            box.setIcon(QtGui.QMessageBox.Information)
            msg = 'Performing this action will save a list of all PDs set for removal\
                    to the <i>outputs/CC</i> directory for future reference.\
                    <br /><br />\
                    Do you want to proceed?'
            box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
            box.setInformativeText(msg)
            reply = box.exec_()

            if reply == QtGui.QMessageBox.Yes:

                trashList = []

                for i in range(1,P3.PrD_total+1):
                    if P4.trashAll[i].isChecked():
                        trashList.append(int(1))
                    else:
                        trashList.append(int(0))

                timestr = time.strftime("%Y%m%d-%H%M%S")
                trashDir = os.path.join(P1.user_directory, 'outputs_%s/CC/temp_removals_%s.txt' % (p.proj_name,timestr))
                np.savetxt(trashDir, trashList, fmt='%i', delimiter='\t')

                box = QtGui.QMessageBox(self)
                box.setWindowTitle('%s Save Current Removals' % progname)
                box.setIcon(QtGui.QMessageBox.Information)
                box.setText('<b>Saving Complete</b>')
                box.setFont(font_standard)
                msg = 'Current removal selections have been saved to the <i>outputs/CC</i> directory.'
                box.setStandardButtons(QtGui.QMessageBox.Ok)
                box.setInformativeText(msg)
                reply = box.exec_()

            elif reply == QtGui.QMessageBox.No:
                pass

    def trashLoad(self):
        self.btn_anchList.setDisabled(True)
        self.btn_anchReset.setDisabled(True)
        self.btn_anchSave.setDisabled(True)
        self.btn_anchLoad.setDisabled(True)
        self.btn_trashList.setDisabled(True)
        self.btn_trashReset.setDisabled(True)
        self.btn_trashSave.setDisabled(True)
        self.btn_trashLoad.setDisabled(True)
        self.btn_occList.setDisabled(True)
        self.btn_rebedList.setDisabled(True)

        P4.trash_list = []
        trash_sum = 0
        for i in range(1,P3.PrD_total+1):
            if P4.trashAll[i].isChecked():
                trash_sum += 1
        if trash_sum == 0:
            fname = QtGui.QFileDialog.getOpenFileName(self, 'Choose Data File', '',
                                     ('Data Files (*.txt)'))[0]
            if fname:
                try:
                    data = []
                    with open(fname) as values:
                        for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                            data.append(column)
                    P4.trash_list = data[0]
                    p.trash_list = P4.trash_list

                    trashLen = 0
                    for i in P4.trash_list:
                        if int(i) == int(1):
                            trashLen += 1

                    idx = 1 #PD index
                    prog = 0
                    PDSeleCanvas1.progBar2.setValue(prog)
                    PDSeleCanvas1.progBar2.setVisible(True)
                    for i in P4.trash_list:
                        if int(i) == int(1): #if PD set to True (remove)
                            P4.entry_PrD.setValue(idx)
                            P4.user_PrD = idx
                            P4.PrD_hist = idx
                            P4.trashAll[idx].setChecked(True)
                            P4.anchorsAll[idx].setChecked(False)
                        prog += (1./trashLen)*100
                        PDSeleCanvas1.progBar2.setValue(prog)
                        idx += 1

                    P4.entry_PrD.setValue(1)
                    PDSeleCanvas1.progBar2.setValue(100)

                    box = QtGui.QMessageBox(self)
                    box.setWindowTitle('%s Load Previous Removals' % progname)
                    box.setIcon(QtGui.QMessageBox.Information)
                    box.setText('<b>Loading Complete</b>')
                    box.setFont(font_standard)
                    msg = 'Previous removal selections have been loaded on the <i>Eigenvectors</i> tab.'
                    box.setStandardButtons(QtGui.QMessageBox.Ok)
                    box.setInformativeText(msg)
                    reply = box.exec_()

                except: #IndexError:
                    box = QtGui.QMessageBox(self)
                    box.setWindowTitle('%s Error' % progname)
                    box.setText('<b>Input Error</b>')
                    box.setIcon(QtGui.QMessageBox.Warning)
                    box.setFont(font_standard)
                    box.setInformativeText('Incorrect file structure detected.')
                    box.setStandardButtons(QtGui.QMessageBox.Ok)
                    box.setDefaultButton(QtGui.QMessageBox.Ok)
                    ret = box.exec_()
            else:
                pass

        elif trash_sum > 0:
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Warning' % progname)
            box.setIcon(QtGui.QMessageBox.Warning)
            box.setText('<b>Input Warning</b>')
            box.setFont(font_standard)
            msg = 'To load PD removals from a previous session, first clear all currently selected\
                    PD removals via the <i>Reset Removals</i> button.'
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setInformativeText(msg)
            reply = box.exec_()

        self.btn_anchList.setDisabled(False)
        self.btn_anchReset.setDisabled(False)
        self.btn_anchSave.setDisabled(False)
        self.btn_anchLoad.setDisabled(False)
        self.btn_trashList.setDisabled(False)
        self.btn_trashReset.setDisabled(False)
        self.btn_trashSave.setDisabled(False)
        self.btn_trashLoad.setDisabled(False)
        self.btn_occList.setDisabled(False)
        self.btn_rebedList.setDisabled(False)

    def occListGen(self):
        sorted_PrDs = sorted(zip(P1.thresh_PrDs,P1.thresh_occ), key=lambda x: x[1], reverse=True)
        self.PrD_table = occTable(data = sorted_PrDs)
        sizeObject = QtGui.QDesktopWidget().screenGeometry(-1) #user screen size
        self.PrD_table.move((sizeObject.width()/2)-100,(sizeObject.height()/2)-300)
        self.PrD_table.show()

    def rebedListGen(self):
        # read points from re-embedding file:
        fname = os.path.join(P1.user_directory,'outputs_{}/topos/Euler_PrD/PrD_embeds.txt'.format(p.proj_name))
        data = []
        with open(fname) as values:
            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                data.append(column)
                
        rebeds0 = data[0]
        rebeds = []
        total = []
        idx = 1
        for i in rebeds0:
            if int(i) == 0:
                rebeds.append('True')
            else:
                rebeds.append('False')
            total.append(idx)
            idx+=1

        if len(rebeds) > 0:
            sorted_rebeds = sorted(zip(total,rebeds), key=lambda x: x[1], reverse=True)
            self.rebed_table = rebedTable(data = sorted_rebeds)
            sizeObject = QtGui.QDesktopWidget().screenGeometry(-1) #user screen size
            self.rebed_table.move((sizeObject.width()/2)-100,(sizeObject.height()/2)-300)
            self.rebed_table.show()
        else:
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Error' % progname)
            box.setText('<b>Input Error</b>')
            box.setFont(font_standard)
            box.setIcon(QtGui.QMessageBox.Information)
            box.setInformativeText('No manifold re-embeddings have been performed.\
                                    <br /><br />\
                                    Manifolds for each PD can be individually re-embedded\
                                    within the <i>View Chosen Topos</i> window.')
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setDefaultButton(QtGui.QMessageBox.Ok)
            ret = box.exec_()


class PDSeleCanvas2(QtGui.QDialog):
    def __init__(self, parent=None):
        super(PDSeleCanvas2, self).__init__(parent)
        self.left = 10
        self.top = 10

        # create canvas and plot data:
        PDSeleCanvas2.figure = Figure(dpi=200)
        PDSeleCanvas2.canvas = FigureCanvas(PDSeleCanvas2.figure)
        PDSeleCanvas2.toolbar = NavigationToolbar(PDSeleCanvas2.canvas, self)
        PDSeleCanvas2.axes = PDSeleCanvas2.figure.add_subplot(1,1,1, polar=True)
        PDSeleAll = PDSeleCanvas2.axes.scatter([0], [0], edgecolor='k', linewidth=.5, c=[0], s=5, cmap=cm.hsv) #empty for init
        #PDSeleCanvas2.axes.autoscale()

        # thetas = [0,45,90,135,180,225,270,315] #in same order as labels below (ref only)
        theta_labels = ['%s180%s'%(u"\u00B1",u"\u00b0"),'-135%s'%(u"\u00b0"),'-90%s'%(u"\u00b0"),'-45%s'%(u"\u00b0"),
                        '0%s'%(u"\u00b0"),'45%s'%(u"\u00b0"),'90%s'%(u"\u00b0"),'135%s'%(u"\u00b0")]
        PDSeleCanvas2.axes.set_ylim(0,180)
        PDSeleCanvas2.axes.set_yticks(np.arange(0,180,20))
        PDSeleCanvas2.axes.set_xticklabels(theta_labels)
        for tick in PDSeleCanvas2.axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in PDSeleCanvas2.axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        PDSeleCanvas2.axes.grid(alpha=0.2)

        layout = QtGui.QGridLayout()
        layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)

        layout.addWidget(PDSeleCanvas2.toolbar, 0, 0, 1, 5, QtCore.Qt.AlignVCenter)
        layout.addWidget(PDSeleCanvas2.canvas, 1, 0, 10, 5, QtCore.Qt.AlignVCenter)

        self.setLayout(layout)


# =============================================================================
# Plot 2d class average:
# =============================================================================

class ClassAvgMain(QtGui.QMainWindow):
    def __init__(self):
        super(ClassAvgMain, self).__init__()
        self.left = 10
        self.top = 10
        self.initUI()
 
    def initUI(self):
        centralwidget = QtGui.QWidget()
        self.setCentralWidget(centralwidget)
        image = ClassAvgCanvas(self, width=2, height=2)
        toolbar =  NavigationToolbar(image, self)
        vbl = QtGui.QVBoxLayout(centralwidget)
        vbl.addWidget(toolbar)
        vbl.addWidget(image)
        self.show()

class ClassAvgCanvas(FigureCanvas):
    def __init__(self, parent=None, width=2, height=2, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.updateGeometry()
        self.plot()
 
    def plot(self):
        fname = os.path.join(P1.user_directory,'outputs_%s/topos/PrD_%s/class_avg.png' % (p.proj_name,P4.user_PrD))
        img = mpimg.imread(fname)
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.set_title('2D Class Average', fontsize=6)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        self.draw()


# =============================================================================
# Individual PrD subwindow:
# =============================================================================

class PrD_Viz(QtGui.QMainWindow):
    def __init__(self):
        super(PrD_Viz, self).__init__()
        self.left = 10
        self.top = 10
        self.initUI()

        '''ZULU?
        # Sub-Help Menu:
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction('&NLSA Movie', self.guide_movie)

    def guide_movie(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>NLSA Movie</b>')
        box.setFont(font_standard)
        box.setInformativeText("<span style='font-weight:normal;'>\
                                line 1\
                                <br /><br />\
                                line 2\
                                <br /><br />\
                                line 3.\
                                </span>")
        box.setStandardButtons(QtGui.QMessageBox.Ok)
        ret = box.exec_()
        '''

    def initUI(self):
        Manifold2dCanvas.eigChoice1 = 0 
        Manifold2dCanvas.eigChoice2 = 1
        Manifold2dCanvas.eigChoice3 = 2
        
        vid_tab1 = VidCanvas(self)
        vid_tab2 = Manifold2dCanvas(self)
        vid_tab3 = Manifold3dCanvas(self)
        vid_tab4 = ChronosCanvas(self)
        vid_tab5 = PsiCanvas(self)
        vid_tab6 = TauCanvas(self)
        global vid_tabs
        vid_tabs = QtGui.QTabWidget(self)
        vid_tabs.addTab(vid_tab1, 'Movie Player')
        vid_tabs.addTab(vid_tab2, '2D Embedding')
        vid_tabs.addTab(vid_tab3, '3D Embedding')
        vid_tabs.addTab(vid_tab4, 'Chronos')
        vid_tabs.addTab(vid_tab5, 'Psi Analysis')
        vid_tabs.addTab(vid_tab6, 'Tau Analysis')
        #vid_tabs.setTabEnabled(1, False)
        vid_tabs.currentChanged.connect(self.onTabChange) #signal for tabs changed via direct click
     
        style = """QTabWidget::tab-bar{
                alignment: center;
                }"""
        self.setStyleSheet(style)
        self.setCentralWidget(vid_tabs)
        #self.setWindowModality(QtCore.Qt.ApplicationModal) #freezes out parent window
        self.show()

    def closeEvent(self, ce): #when user clicks to exit via subwindow button
        VidCanvas.f = 0
        VidCanvas.run = 0 #needed to pause scrollbar before it is deleted
        VidCanvas.canvas.stop_event_loop()

        if int(0) < Manifold2dCanvas.progress1.value() < int(100): #no escaping mid-thread
            ce.ignore()

    def onTabChange(self, i):
        if i != 0: #needed to stop `Movie Player` if tab changed during playback
            VidCanvas.run = 0
            VidCanvas.canvas.stop_event_loop()
            VidCanvas.buttonForward.setDisabled(False)
            VidCanvas.buttonForward.setFocus()
            VidCanvas.buttonBackward.setDisabled(False)
            VidCanvas.buttonPause.setDisabled(True)


################################################################################
# individual PrD subwindow:

class PrD2_Viz(QtGui.QMainWindow):
    def __init__(self):
        super(PrD2_Viz, self).__init__()
        self.left = 10
        self.top = 10
        self.initUI()

        ''' ZULU?
        # Sub-Help Menu:
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction('&NLSA Movie', self.guide_movie)

    def guide_movie(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>NLSA Movie</b>')
        box.setFont(font_standard)
        box.setInformativeText("<span style='font-weight:normal;'>\
                                line 1\
                                <br /><br />\
                                line 2\
                                <br /><br />\
                                line 3.\
                                </span>")
        box.setStandardButtons(QtGui.QMessageBox.Ok)
        ret = box.exec_()
        '''

    def initUI(self):
        centralwidget = QtGui.QWidget()
        self.setCentralWidget(centralwidget)

        vid2_movs = Vid2Canvas(self)

        layout = QtGui.QGridLayout(centralwidget)
        layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        layout.addWidget(vid2_movs, 0,0,1,1)

        self.setWindowModality(QtCore.Qt.ApplicationModal) #freezes out parent window
        self.show()

    def closeEvent(self, ce): #when user clicks to exit via subwindow button
        Vid2Canvas.f = 0
        Vid2Canvas.run = 0 #needed to pause scrollbar before it is deleted
        Vid2Canvas.canvas1.stop_event_loop()
        Vid2Canvas.canvas2.stop_event_loop()


class VidCanvas(QtGui.QDialog):
    imgDir = ''
    img_paths = []
    imgs = []
    frames = 0 #total number of frames
    run = 0 #switch, {-1,0,1} :: {backwards,pause,forward}
    f = 0 #frame index (current frame)
    rec = 0 #safeguard for recursion limit
    delay = .001 #playback delay in ms

    def __init__(self, parent=None):
        super(VidCanvas, self).__init__(parent)

        i = 0
        for root, dirs, files in os.walk(P4.vidDir):
            for file in sorted(files):
                if not file.startswith('.'): #ignore hidden files
                    if file.endswith(".png"):
                        VidCanvas.img_paths.append(os.path.join(root, file))
                        VidCanvas.imgs.append(imageio.imread(VidCanvas.img_paths[i]))
                        i += 1

        VidCanvas.frames = int(len(VidCanvas.imgs)) - 1

        self.figure = Figure(dpi=200, facecolor='w', edgecolor='w')
        self.ax = self.figure.add_axes([0,0,1,1])
        self.ax.axis('off')
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)

        for item in [self.figure, self.ax]:
            item.patch.set_visible(False)

        VidCanvas.canvas = FigureCanvas(self.figure)
        #self.toolbar = NavigationToolbar(VidCanvas.canvas, self)
        self.currentIMG = self.ax.imshow(VidCanvas.imgs[0]) #plot initial data
        VidCanvas.canvas.draw() #refresh canvas

        # player control buttons:
        self.buttonF1 = QtGui.QPushButton(u'\u21E5')
        self.buttonF1.clicked.connect(self.F1)
        self.buttonF1.setDisabled(False)
        self.buttonF1.setDefault(False)
        self.buttonF1.setAutoDefault(False)

        VidCanvas.buttonForward = QtGui.QPushButton(u'\u25B6')
        VidCanvas.buttonForward.clicked.connect(self.forward)
        VidCanvas.buttonForward.setDisabled(False)
        VidCanvas.buttonForward.setDefault(True)
        VidCanvas.buttonForward.setAutoDefault(True)

        VidCanvas.buttonPause = QtGui.QPushButton(u'\u25FC')
        VidCanvas.buttonPause.clicked.connect(self.pause)
        VidCanvas.buttonPause.setDisabled(True)
        VidCanvas.buttonPause.setDefault(False)
        VidCanvas.buttonPause.setAutoDefault(False)

        VidCanvas.buttonBackward = QtGui.QPushButton(u'\u25C0')
        VidCanvas.buttonBackward.clicked.connect(self.backward)
        VidCanvas.buttonBackward.setDisabled(False)
        VidCanvas.buttonBackward.setDefault(False)
        VidCanvas.buttonBackward.setAutoDefault(False)

        self.buttonB1 = QtGui.QPushButton(u'\u21E4')
        self.buttonB1.clicked.connect(self.B1)
        self.buttonB1.setDisabled(False)
        self.buttonB1.setDefault(False)
        self.buttonB1.setAutoDefault(False)

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(VidCanvas.frames)
        self.slider.setValue(0)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.sliderUpdate)

        # create layout:
        layout = QtGui.QGridLayout()
        layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        #layout.addWidget(self.toolbar, 0,0,1,5)
        layout.addWidget(VidCanvas.canvas, 2,0,1,5)
        layout.addWidget(self.buttonB1, 3,0,1,1)
        layout.addWidget(VidCanvas.buttonBackward, 3,1,1,1)
        layout.addWidget(VidCanvas.buttonPause, 3,2,1,1)
        layout.addWidget(VidCanvas.buttonForward, 3,3,1,1)
        layout.addWidget(self.buttonF1, 3,4,1,1)
        layout.addWidget(self.slider,4,0,1,5)
        self.setLayout(layout)

    def scroll(self, frame):
        VidCanvas.canvas.stop_event_loop()

        self.slider.setValue(VidCanvas.f)
        self.currentIMG.set_data(VidCanvas.imgs[VidCanvas.f]) #update data
        VidCanvas.canvas.draw() #refresh canvas
        VidCanvas.canvas.start_event_loop(self.delay)

        if VidCanvas.run == 1:
            if VidCanvas.f < VidCanvas.frames:
                VidCanvas.f += 1
                self.scroll(VidCanvas.f)
            elif VidCanvas.f == VidCanvas.frames:
                VidCanvas.f = 0
                self.rec +=1 #recursion safeguard
                if self.rec == 10:
                    self.rec = 0
                    self.pause()
                else:
                    self.scroll(VidCanvas.f)
        elif VidCanvas.run == -1:
            if VidCanvas.f > 0:
                VidCanvas.f -= 1
                self.scroll(VidCanvas.f)
            elif VidCanvas.f == 0:
                VidCanvas.f = VidCanvas.frames
                self.rec +=1 #recusion safeguard
                if self.rec == 10:
                    self.rec = 0
                    self.pause()
                else:
                    self.scroll(VidCanvas.f)
        elif VidCanvas.run == 0:
            VidCanvas.canvas.stop_event_loop()

    def F1(self): #forward one frame
        VidCanvas.buttonPause.setDisabled(True)
        VidCanvas.buttonForward.setDisabled(False)
        VidCanvas.buttonBackward.setDisabled(False)
        self.buttonF1.setFocus()

        VidCanvas.run = 0
        self.rec = 0
        VidCanvas.canvas.stop_event_loop()
        if VidCanvas.f == VidCanvas.frames:
            VidCanvas.f = 0
        else:
            VidCanvas.f += 1

        self.slider.setValue(VidCanvas.f)
        self.currentIMG.set_data(VidCanvas.imgs[VidCanvas.f]) #update data
        VidCanvas.canvas.draw() #refresh canvas


    def forward(self): #play forward
        VidCanvas.buttonForward.setDisabled(True)
        VidCanvas.buttonBackward.setDisabled(False)
        VidCanvas.buttonPause.setDisabled(False)
        VidCanvas.buttonPause.setFocus()

        VidCanvas.run = 1
        self.rec = 0
        self.scroll(VidCanvas.f)

    def pause(self): #stop play
        VidCanvas.buttonForward.setDisabled(False)
        VidCanvas.buttonBackward.setDisabled(False)
        VidCanvas.buttonPause.setDisabled(True)
        VidCanvas.buttonForward.setFocus()

        VidCanvas.run = 0
        self.rec = 0
        self.scroll(VidCanvas.f)

    def backward(self): #play backward          
        VidCanvas.buttonBackward.setDisabled(True)
        VidCanvas.buttonForward.setDisabled(False)
        VidCanvas.buttonPause.setDisabled(False)
        VidCanvas.buttonPause.setFocus()

        VidCanvas.run = -1
        self.rec = 0
        self.scroll(VidCanvas.f)

    def B1(self): #backward one frame
        VidCanvas.buttonPause.setDisabled(True)
        VidCanvas.buttonForward.setDisabled(False)
        VidCanvas.buttonBackward.setDisabled(False)
        self.buttonB1.setFocus()

        VidCanvas.run = 0
        self.rec = 0
        VidCanvas.canvas.stop_event_loop()
        if VidCanvas.f == 0:
            VidCanvas.f = VidCanvas.frames
        else:
            VidCanvas.f -= 1

        self.slider.setValue(VidCanvas.f)
        self.currentIMG.set_data(VidCanvas.imgs[VidCanvas.f]) #update data
        VidCanvas.canvas.draw() #refresh canvas

    def sliderUpdate(self): #update frame based on user slider position       
        if VidCanvas.f != self.slider.value(): #only if user moves slider position manually
            VidCanvas.buttonPause.setDisabled(True)
            VidCanvas.buttonForward.setDisabled(False)
            VidCanvas.buttonBackward.setDisabled(False)
            VidCanvas.run = 0
            self.rec = 0
            VidCanvas.canvas.stop_event_loop()

            VidCanvas.f = self.slider.value()
            self.currentIMG.set_data(VidCanvas.imgs[self.slider.value()]) #update data
            VidCanvas.canvas.draw() #refresh canvas


class Vid2Canvas(QtGui.QDialog):
    vidDir1 = ''
    vidDir2 = ''
    imgDir1 = ''
    imgDir2 = ''
    img_paths1 = []
    img_paths2 = []
    imgs1 = []
    imgs2 = []
    frames1 = 0 #total number of frames in movie 1
    frames2 = 0 #total number of frames in movie 2
    run = 0 #switch, {-1,0,1} :: {backwards,pause,forward}
    f = 0 #frame index (current frame)
    rec = 0 #safeguard for recursion limit
    delay = .001 #playback delay in ms
    blank = []

    def __init__(self, parent=None):
        super(Vid2Canvas, self).__init__(parent)
        # =====================================================================
        # Create blank image for initiation:
        # =====================================================================
        picDir = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/topos_1.png' % (p.proj_name,P4.user_PrD))
        picImg = Image.open(picDir)
        picSize = picImg.size
        Vid2Canvas.blank = np.ones([picSize[0],picSize[1],3], dtype=int)*255 #white background
        # =====================================================================
        self.vidDir1 = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/psi_%s/' % (p.proj_name,1,1))
        i = 0
        for root, dirs, files in os.walk(self.vidDir1):
            for file in sorted(files):
                if not file.startswith('.'): #ignore hidden files
                    if file.endswith(".png"):
                        self.imgs1.append(Vid2Canvas.blank)
                        self.imgs2.append(Vid2Canvas.blank)
                        i += 1

        self.label_Vline = QtGui.QLabel('') #separating line
        self.label_Vline.setFont(font_standard)
        self.label_Vline.setFrameStyle(QtGui.QFrame.VLine | QtGui.QFrame.Sunken)

        self.label_mov1 = QtGui.QLabel('NLSA Movie 1')
        self.label_mov1.setFont(font_standard)
        self.label_mov1.setMargin(15)
        self.label_mov1.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.PrD1 = QtGui.QSpinBox(self)
        self.PrD1.setMinimum(1)
        self.PrD1.setMaximum(P3.PrD_total)
        self.PrD1.setFont(font_standard)
        self.PrD1.setPrefix('PD: ')
        self.PrD1.setDisabled(False)

        self.Psi1 = QtGui.QSpinBox(self)
        self.Psi1.setMinimum(1)
        self.Psi1.setMaximum(p.num_psis)
        self.Psi1.setFont(font_standard)
        self.Psi1.setPrefix('Psi: ')
        self.Psi1.setDisabled(False)

        self.sense1 = QtGui.QComboBox(self)
        self.sense1.addItem('Sense: FWD')
        self.sense1.addItem('Sense: REV')
        self.sense1.setFont(font_standard)
        self.sense1.setToolTip('Sense for selected movie.')
        self.sense1.setDisabled(False)

        self.btnSet1 = QtGui.QCheckBox('Set Movie 1')
        self.btnSet1.clicked.connect(self.setMovie1)
        self.btnSet1.setChecked(False)
        self.btnSet1.setDisabled(False)

        self.label_mov2 = QtGui.QLabel('NLSA Movie 2')
        self.label_mov2.setFont(font_standard)
        self.label_mov2.setMargin(15)
        self.label_mov2.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.PrD2 = QtGui.QSpinBox(self)
        self.PrD2.setMinimum(1)
        self.PrD2.setMaximum(P3.PrD_total)
        self.PrD2.setFont(font_standard)
        self.PrD2.setPrefix('PD: ')
        self.PrD2.setDisabled(False)

        self.Psi2 = QtGui.QSpinBox(self)
        self.Psi2.setMinimum(1)
        self.Psi2.setMaximum(p.num_psis)
        self.Psi2.setFont(font_standard)
        self.Psi2.setPrefix('Psi: ')
        self.Psi2.setDisabled(False)

        self.sense2 = QtGui.QComboBox(self)
        self.sense2.addItem('Sense: FWD')
        self.sense2.addItem('Sense: REV')
        self.sense2.setFont(font_standard)
        self.sense2.setToolTip('Sense for selected movie.')
        self.sense2.setDisabled(False)

        self.btnSet2 = QtGui.QCheckBox('Set Movie 2')
        self.btnSet2.clicked.connect(self.setMovie2)
        self.btnSet2.setChecked(False)
        self.btnSet2.setDisabled(False)

        self.frames1 = int(len(self.imgs1)) - 1

        self.figure1 = Figure(dpi=200, facecolor='w', edgecolor='w')
        self.ax1 = self.figure1.add_axes([0,0,1,1])
        self.ax1.axis('off')
        self.ax1.xaxis.set_visible(False)
        self.ax1.yaxis.set_visible(False)

        for item in [self.figure1, self.ax1]:
            item.patch.set_visible(False)

        Vid2Canvas.canvas1 = FigureCanvas(self.figure1)
        self.currentIMG1 = self.ax1.imshow(self.imgs1[0]) #plot initial data
        Vid2Canvas.canvas1.draw() #refresh canvas

        self.frames2 = int(len(self.imgs2)) - 1

        self.figure2 = Figure(dpi=200, facecolor='w', edgecolor='w')
        self.ax2 = self.figure2.add_axes([0,0,1,1])
        self.ax2.axis('off')
        self.ax2.xaxis.set_visible(False)
        self.ax2.yaxis.set_visible(False)

        for item in [self.figure2, self.ax2]:
            item.patch.set_visible(False)

        Vid2Canvas.canvas2 = FigureCanvas(self.figure2)
        self.currentIMG2 = self.ax2.imshow(self.imgs2[0]) #plot initial data
        Vid2Canvas.canvas2.draw() #refresh canvas

        # player control buttons:
        self.buttonF1 = QtGui.QPushButton(u'\u21E5')
        self.buttonF1.clicked.connect(self.F1)
        self.buttonF1.setDisabled(False)
        self.buttonF1.setDefault(False)
        self.buttonF1.setAutoDefault(False)

        Vid2Canvas.buttonForward = QtGui.QPushButton(u'\u25B6')
        Vid2Canvas.buttonForward.clicked.connect(self.forward)
        Vid2Canvas.buttonForward.setDisabled(False)
        Vid2Canvas.buttonForward.setDefault(True)
        Vid2Canvas.buttonForward.setAutoDefault(True)

        Vid2Canvas.buttonPause = QtGui.QPushButton(u'\u25FC')
        Vid2Canvas.buttonPause.clicked.connect(self.pause)
        Vid2Canvas.buttonPause.setDisabled(True)
        Vid2Canvas.buttonPause.setDefault(False)
        Vid2Canvas.buttonPause.setAutoDefault(False)

        Vid2Canvas.buttonBackward = QtGui.QPushButton(u'\u25C0')
        Vid2Canvas.buttonBackward.clicked.connect(self.backward)
        Vid2Canvas.buttonBackward.setDisabled(False)
        Vid2Canvas.buttonBackward.setDefault(False)
        Vid2Canvas.buttonBackward.setAutoDefault(False)

        self.buttonB1 = QtGui.QPushButton(u'\u21E4')
        self.buttonB1.clicked.connect(self.B1)
        self.buttonB1.setDisabled(False)
        self.buttonB1.setDefault(False)
        self.buttonB1.setAutoDefault(False)

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.frames1)
        self.slider.setValue(0)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.sliderUpdate)

        # create layout:
        layout = QtGui.QGridLayout()
        layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)

        layout.addWidget(self.label_mov1, 0,2,1,2)
        layout.addWidget(self.label_mov2, 0,7,1,2)

        layout.addWidget(self.PrD1, 1,1,1,1)
        layout.addWidget(self.Psi1, 1,2,1,1)
        layout.addWidget(self.sense1, 1,3,1,1)
        layout.addWidget(self.btnSet1, 1,4,1,1)

        layout.addWidget(self.PrD2, 1,6,1,1)
        layout.addWidget(self.Psi2, 1,7,1,1)
        layout.addWidget(self.sense2, 1,8,1,1)
        layout.addWidget(self.btnSet2, 1,9,1,1)

        layout.addWidget(Vid2Canvas.canvas1, 2,1,1,4)
        layout.addWidget(Vid2Canvas.canvas2, 2,6,1,4)

        layout.addWidget(self.buttonB1, 5,2,1,1)
        layout.addWidget(Vid2Canvas.buttonBackward, 5,3,1,1)
        layout.addWidget(Vid2Canvas.buttonPause, 5,4,1,3)
        layout.addWidget(Vid2Canvas.buttonForward, 5,7,1,1)
        layout.addWidget(self.buttonF1, 5,8,1,1)
        layout.addWidget(self.slider,6,2,1,7)

        layout.addWidget(self.label_Vline, 0, 5, 5, 1)

        self.setLayout(layout)

    def setMovie1(self):
        Vid2Canvas.f = 0
        Vid2Canvas.run = 0
        Vid2Canvas.rec = 0
        Vid2Canvas.canvas1.stop_event_loop()
        Vid2Canvas.canvas2.stop_event_loop()
        Vid2Canvas.buttonForward.setDisabled(False)
        Vid2Canvas.buttonForward.setFocus()
        Vid2Canvas.buttonBackward.setDisabled(False)
        Vid2Canvas.buttonPause.setDisabled(True)
        self.img_paths1 = []
        self.imgs1 = []
        self.vidDir1 = ''

        prD = self.PrD1.value()
        psi = self.Psi1.value()
        self.vidDir1 = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/psi_%s/' % (p.proj_name,prD,psi))

        if self.btnSet1.isChecked():
            self.PrD1.setDisabled(True)
            self.Psi1.setDisabled(True)
            self.sense1.setDisabled(True)

            i = 0
            for root, dirs, files in os.walk(self.vidDir1):
                for file in sorted(files):
                    if not file.startswith('.'): #ignore hidden files
                        if file.endswith(".png"):
                            self.img_paths1.append(os.path.join(root, file))
                            self.imgs1.append(imageio.imread(self.img_paths1[i]))
                            i += 1

            if self.sense1.currentText() == 'Sense: REV':
                self.imgs1.reverse()

        else:
            self.PrD1.setDisabled(False)
            self.Psi1.setDisabled(False)
            self.sense1.setDisabled(False)

            i = 0
            for root, dirs, files in os.walk(self.vidDir1):
                for file in sorted(files):
                    if not file.startswith('.'): #ignore hidden files
                        if file.endswith(".png"):
                            self.imgs1.append(Vid2Canvas.blank)
                            i += 1

        gc.collect()
        Vid2Canvas.canvas1.flush_events()

        self.frames1 = int(len(self.imgs1)) - 1
        self.slider.setMaximum(self.frames1)
        self.currentIMG1 = self.ax1.imshow(self.imgs1[0]) #plot initial frame
        self.slider.setValue(0)
        self.f = self.slider.value()

        Vid2Canvas.canvas1.draw() #refresh canvas 1


    def setMovie2(self):
        Vid2Canvas.f = 0
        Vid2Canvas.run = 0
        Vid2Canvas.rec = 0
        Vid2Canvas.canvas1.stop_event_loop()
        Vid2Canvas.canvas2.stop_event_loop()
        Vid2Canvas.buttonForward.setDisabled(False)
        Vid2Canvas.buttonForward.setFocus()
        Vid2Canvas.buttonBackward.setDisabled(False)
        Vid2Canvas.buttonPause.setDisabled(True)
        self.img_paths2 = []
        self.imgs2 = []
        self.vidDir2 = ''

        prD = self.PrD2.value()
        psi = self.Psi2.value()
        self.vidDir2 = os.path.join(P1.user_directory, 'outputs_%s/topos/PrD_%s/psi_%s/' % (p.proj_name,prD,psi))

        if self.btnSet2.isChecked():
            self.PrD2.setDisabled(True)
            self.Psi2.setDisabled(True)
            self.sense2.setDisabled(True)

            i = 0
            for root, dirs, files in os.walk(self.vidDir2):
                for file in sorted(files):
                    if not file.startswith('.'): #ignore hidden files
                        if file.endswith(".png"):
                            self.img_paths2.append(os.path.join(root, file))
                            self.imgs2.append(imageio.imread(self.img_paths2[i]))
                            i += 1

            if self.sense2.currentText() == 'Sense: REV':
                self.imgs2.reverse()

        else:
            self.PrD2.setDisabled(False)
            self.Psi2.setDisabled(False)
            self.sense2.setDisabled(False)

            i = 0
            for root, dirs, files in os.walk(self.vidDir2):
                for file in sorted(files):
                    if not file.startswith('.'): #ignore hidden files
                        if file.endswith(".png"):
                            self.imgs2.append(Vid2Canvas.blank)
                            i += 1
            
        gc.collect()
        Vid2Canvas.canvas2.flush_events()

        self.frames2 = int(len(self.imgs2)) - 1
        self.slider.setMaximum(self.frames2)
        self.currentIMG2 = self.ax2.imshow(self.imgs2[0]) #plot initial frame
        self.slider.setValue(0)
        self.f = self.slider.value()

        Vid2Canvas.canvas2.draw() #refresh canvas 2


    def scroll(self, frame):
        Vid2Canvas.canvas1.stop_event_loop()
        Vid2Canvas.canvas2.stop_event_loop()

        self.slider.setValue(self.f)
        self.currentIMG1.set_data(self.imgs1[self.f]) #update data 1
        self.currentIMG2.set_data(self.imgs2[self.f]) #update data 2
        Vid2Canvas.canvas1.draw() #refresh canvas 1
        Vid2Canvas.canvas2.draw() #refresh canvas 2
        Vid2Canvas.canvas1.start_event_loop(self.delay)
        Vid2Canvas.canvas2.start_event_loop(self.delay)

        if self.run == 1:
            if self.f < self.frames1:
                self.f += 1
                self.scroll(self.f)
            elif self.f == self.frames1:
                self.f = 0
                self.rec +=1 #recursion safeguard
                if self.rec == 10:
                    self.rec = 0
                    self.pause()
                else:
                    self.scroll(self.f)
        elif self.run == -1:
            if self.f > 0:
                self.f -= 1
                self.scroll(self.f)
            elif self.f == 0:
                self.f = self.frames1
                self.rec +=1 #recusion safeguard
                if self.rec == 10:
                    self.rec = 0
                    self.pause()
                else:
                    self.scroll(self.f)
        elif self.run == 0:
            Vid2Canvas.canvas1.stop_event_loop()
            Vid2Canvas.canvas2.stop_event_loop()

    def F1(self): #forward one frame
        Vid2Canvas.buttonPause.setDisabled(True)
        Vid2Canvas.buttonForward.setDisabled(False)
        Vid2Canvas.buttonBackward.setDisabled(False)
        self.buttonF1.setFocus()

        self.run = 0
        self.rec = 0
        Vid2Canvas.canvas1.stop_event_loop()
        Vid2Canvas.canvas2.stop_event_loop()
        if self.f == self.frames1:
            self.f = 0
        else:
            self.f += 1

        self.slider.setValue(self.f)
        self.currentIMG1.set_data(self.imgs1[self.f]) #update data 1
        self.currentIMG2.set_data(self.imgs2[self.f]) #update data 2
        Vid2Canvas.canvas1.draw() #refresh canvas 1
        Vid2Canvas.canvas2.draw() #refresh canvas 2


    def forward(self): #play forward
        Vid2Canvas.buttonForward.setDisabled(True)
        Vid2Canvas.buttonBackward.setDisabled(False)
        Vid2Canvas.buttonPause.setDisabled(False)
        Vid2Canvas.buttonPause.setFocus()

        self.run = 1
        self.rec = 0
        self.scroll(self.f)

    def pause(self): #stop play
        Vid2Canvas.buttonForward.setDisabled(False)
        Vid2Canvas.buttonBackward.setDisabled(False)
        Vid2Canvas.buttonPause.setDisabled(True)
        Vid2Canvas.buttonForward.setFocus()

        self.run = 0
        self.rec = 0
        self.scroll(self.f)

    def backward(self): #play backward
        Vid2Canvas.buttonBackward.setDisabled(True)
        Vid2Canvas.buttonForward.setDisabled(False)
        Vid2Canvas.buttonPause.setDisabled(False)
        Vid2Canvas.buttonPause.setFocus()

        self.run = -1
        self.rec = 0
        self.scroll(self.f)

    def B1(self): #backward one frame
        Vid2Canvas.buttonPause.setDisabled(True)
        Vid2Canvas.buttonForward.setDisabled(False)
        Vid2Canvas.buttonBackward.setDisabled(False)
        self.buttonB1.setFocus()

        self.run = 0
        self.rec = 0
        Vid2Canvas.canvas1.stop_event_loop()
        Vid2Canvas.canvas2.stop_event_loop()
        if self.f == 0:
            self.f = self.frames1
        else:
            self.f -= 1

        self.slider.setValue(self.f)
        self.currentIMG1.set_data(self.imgs1[self.f]) #update data 1
        self.currentIMG2.set_data(self.imgs2[self.f]) #update data 2
        Vid2Canvas.canvas1.draw() #refresh canvas 1
        Vid2Canvas.canvas2.draw() #refresh canvas 2


    def sliderUpdate(self): #update frame based on user slider position
        if self.f != self.slider.value(): #only if user moves slider position manually
            Vid2Canvas.buttonPause.setDisabled(True)
            Vid2Canvas.buttonForward.setDisabled(False)
            Vid2Canvas.buttonBackward.setDisabled(False)
            self.run = 0
            self.rec = 0
            Vid2Canvas.canvas1.stop_event_loop()
            Vid2Canvas.canvas2.stop_event_loop()

            self.f = self.slider.value()
            self.currentIMG1.set_data(self.imgs1[self.slider.value()]) #update data 1
            self.currentIMG2.set_data(self.imgs2[self.slider.value()]) #update data 2
            Vid2Canvas.canvas1.draw() #refresh canvas 1
            Vid2Canvas.canvas2.draw() #refresh canvas 2


class Manifold2dCanvas(QtGui.QDialog):
    # for eigenvector specific plots:
    eig_current = 1
    eig_compare1 = 2
    eig_compare2 = 3
    # for changing views based on 3D embedding coordinates chosen:
    eigChoice1 = 0 
    eigChoice2 = 1
    eigChoice3 = 2
    
    coordsX = [] #user X coordinate picks
    coordsY = [] #user Y coordinate picks
    connected = 0 #binary: 0=unconnected, 1=connected
    pts_orig = []
    pts_origX = []
    pts_origY = []
    pts_new = []
    pts_newX = []
    pts_newY = []
    pts_encircled = []
    pts_encircledX = []
    pts_encircledY = []
    x = []
    y = []
    imgAvg = []
    progress1Changed = QtCore.Signal(int)
    progress2Changed = QtCore.Signal(int)

    def __init__(self, parent=None):
        super(Manifold2dCanvas, self).__init__(parent)

        self.figure = Figure(dpi=200)
        self.ax = self.figure.add_subplot(111)
        #self.ax.set_aspect('equal')
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        #self.toolbar = NavigationToolbar(self.canvas, self)

        psi_file = '{}prD_{}'.format(p.psi_file, P4.user_PrD - 1) #current embedding
        data = myio.fin1(psi_file)
        x = data['psi'][:,Manifold2dCanvas.eigChoice1]
        y = data['psi'][:,Manifold2dCanvas.eigChoice2]

        Manifold2dCanvas.pts_orig = zip(x,y)
        Manifold2dCanvas.pts_origX = x
        Manifold2dCanvas.pts_origY = y
        self.ax.scatter(Manifold2dCanvas.pts_origX, Manifold2dCanvas.pts_origY, s=1, c='#1f77b4') #plot initial data, C0

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_title('Place points on the plot to encircle deviant cluster(s)', fontsize=3.5)
        self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (Manifold2dCanvas.eigChoice1+1), fontsize=6)
        self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (Manifold2dCanvas.eigChoice2+1), fontsize=6)
        self.ax.autoscale()
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.draw() #refresh canvas

        # canvas buttons:
        Manifold2dCanvas.btn_reset = QtGui.QPushButton('Reset Plot')
        Manifold2dCanvas.btn_reset.clicked.connect(self.reset)
        Manifold2dCanvas.btn_reset.setDisabled(False)
        Manifold2dCanvas.btn_reset.setDefault(False)
        Manifold2dCanvas.btn_reset.setAutoDefault(False)

        self.btn_connect = QtGui.QPushButton('Connect Path')
        self.btn_connect.clicked.connect(self.connect)
        self.btn_connect.setDisabled(True)
        self.btn_connect.setDefault(False)
        self.btn_connect.setAutoDefault(False)

        self.btn_remove = QtGui.QPushButton('Remove Cluster')
        self.btn_remove.clicked.connect(self.remove)
        self.btn_remove.setDisabled(True)
        self.btn_remove.setDefault(False)
        self.btn_remove.setAutoDefault(False)

        self.btn_rebed = QtGui.QPushButton('Update Manifold', self)
        self.btn_rebed.clicked.connect(self.rebed)
        self.btn_rebed.setDisabled(True)
        self.btn_rebed.setDefault(False)
        self.btn_rebed.setAutoDefault(False)

        self.btn_revert = QtGui.QPushButton('Revert Manifold', self)
        self.btn_revert.clicked.connect(self.revert)
        self.btn_revert.setDefault(False)
        self.btn_revert.setAutoDefault(False)
        if P4.origEmbed[P4.user_PrD - 1] == 1: #manifold hasn't been re-embedded
            self.btn_revert.setDisabled(True)
        else:
            self.btn_revert.setDisabled(False)

        self.btn_view = QtGui.QPushButton('View Cluster')
        self.btn_view.clicked.connect(self.view)
        self.btn_view.setDisabled(True)
        self.btn_view.setDefault(False)
        self.btn_view.setAutoDefault(False)

        layout = QtGui.QGridLayout()
        layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        #layout.addWidget(self.toolbar, 0,0,1,5)
        layout.addWidget(self.canvas, 1,0,1,6)
        layout.addWidget(Manifold2dCanvas.btn_reset, 2,0,1,1)
        layout.addWidget(self.btn_connect, 2,1,1,1)
        layout.addWidget(self.btn_view, 2,2,1,1)
        layout.addWidget(self.btn_remove, 2,3,1,1)
        layout.addWidget(self.btn_rebed, 2,4,1,1)
        layout.addWidget(self.btn_revert, 2,5,1,1)

        Manifold2dCanvas.progress1 = QtGui.QProgressBar(minimum=0, maximum=100, value=0)
        layout.addWidget(Manifold2dCanvas.progress1, 3, 0, 1, 6)
        Manifold2dCanvas.progress1.show()

        self.progress1Changed.connect(self.on_progress1Changed)
        self.progress2Changed.connect(self.on_progress2Changed)

        self.setLayout(layout)

    def reset(self):
        if len(self.ax.lines) != 0:
            if self.connected == 0:
                for i in range(1,(len(self.coordsX)*2)):
                    del(self.ax.lines[-1]) #delete all vertices and edges

            elif self.connected == 1:
                for i in range(1,(len(self.coordsX)*2)+1):
                    del(self.ax.lines[-1]) #delete all vertices and edges
                self.connected = 0
        else:
            self.connected = 0

        self.btn_connect.setDisabled(True)
        self.btn_remove.setDisabled(True)
        self.btn_view.setDisabled(True)
        self.btn_rebed.setDisabled(True)
        self.coordsX = []
        self.coordsY = []

        # redraw and resize figure:
        self.ax.clear()
        self.ax.scatter(Manifold2dCanvas.pts_origX, Manifold2dCanvas.pts_origY, s=1, c='#1f77b4')

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_title('Place points on the plot to encircle deviant cluster(s)', fontsize=3.5)
        self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (Manifold2dCanvas.eigChoice1+1), fontsize=6)
        self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (Manifold2dCanvas.eigChoice2+1), fontsize=6)
        self.ax.autoscale()
        self.canvas.draw()

    def connect(self):
        if len(self.coordsX) > 2:
            ax = self.figure.axes[0]
            ax.plot([self.coordsX[0],self.coordsX[-1]],
                    [self.coordsY[0],self.coordsY[-1]],
                    color='#7f7f7f', linestyle='solid', linewidth=.5, zorder=1) #C7
            self.canvas.draw()
        self.connected = 1
        self.btn_connect.setDisabled(True)
        self.btn_remove.setDisabled(False)
        self.btn_view.setDisabled(False)

    def remove(self):
        # reset cropped points if re-clicked:
        Manifold2dCanvas.pts_new = [] 
        Manifold2dCanvas.pts_newX = []
        Manifold2dCanvas.pts_newY = []

        codes = []
        for i in range(len(self.coordsX)):
            if i == 0:
                codes.extend([pltPath.Path.MOVETO])
            elif i == len(self.coordsX):
                codes.extend([pltPath.Path.CLOSEPOLY])
            else:
                codes.extend([pltPath.Path.LINETO])
                
        path = pltPath.Path(list(map(list, zip(self.coordsX, self.coordsY))), codes)
        inside = path.contains_points(np.dstack((Manifold2dCanvas.pts_origX,Manifold2dCanvas.pts_origY))[0].tolist(), radius=1e-9)

        sums = 0 #number of points within polygon
        index = 0
        for i in inside:
            index += 1
            if i == False:
                Manifold2dCanvas.pts_newX.append(Manifold2dCanvas.pts_origX[index-1])
                Manifold2dCanvas.pts_newY.append(Manifold2dCanvas.pts_origY[index-1])
                Manifold2dCanvas.pts_new = zip(Manifold2dCanvas.pts_newX, Manifold2dCanvas.pts_newY)
            else:
                sums += 1 #number of encircled points

        #delta = len(Manifold2dCanvas.pts_newX)

        #if delta >= p.PDsizeThL:
        # crop out points, redraw and resize figure:
        self.ax.clear()
        self.ax.scatter(Manifold2dCanvas.pts_newX, Manifold2dCanvas.pts_newY, s=1, c='#1f77b4')
        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_title('Place points on the plot to encircle deviant cluster(s)', fontsize=3.5)
        self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (Manifold2dCanvas.eigChoice1+1), fontsize=6)
        self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (Manifold2dCanvas.eigChoice2+1), fontsize=6)
        self.ax.autoscale()
        self.canvas.draw()
        #self.connected = 0 #if commented out, no more mouse clicks accepted (until 'reset')
        self.btn_remove.setDisabled(True)
        self.btn_view.setDisabled(True)
        self.btn_rebed.setDisabled(False)

        '''else:
            Manifold2dCanvas.pts_new = []
            Manifold2dCanvas.pts_newX = []
            Manifold2dCanvas.pts_newY = []
            msg = 'The number of remaining points (%s) will be less than\
                    the low threshold limit for each PD (%s).\
                    <br /><br />\
                    Please encircle at least %s fewer points to proceed.' % (delta, int(p.PDsizeThL), int(p.PDsizeThL-delta))
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Error' % progname)
            box.setText('<b>Input Error</b>')
            box.setIcon(QtGui.QMessageBox.Warning)
            box.setFont(font_standard)
            box.setInformativeText(msg)
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setDefaultButton(QtGui.QMessageBox.Ok)
            ret = box.exec_()'''

    def rebed(self):
        msg = 'Performing this action will recalculate the manifold \
                embedding step for the current PD to include only the points shown.\
                <br /><br />\
                Do you want to proceed?'
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s' % progname)
        box.setText('<b>Update Manifold</b>')
        box.setFont(font_standard)
        box.setIcon(QtGui.QMessageBox.Question)
        box.setInformativeText(msg)
        box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
        reply = box.exec_()
        if reply == QtGui.QMessageBox.Yes:
            Manifold2dCanvas.btn_reset.setDisabled(True)
            self.btn_rebed.setDisabled(True)
            vid_tabs.setTabEnabled(0, False)
            vid_tabs.setTabEnabled(2, False)
            vid_tabs.setTabEnabled(3, False)
            vid_tabs.setTabEnabled(4, False)
            vid_tabs.setTabEnabled(5, False)

            if P4.origEmbed[P4.user_PrD - 1] == 1: #only make a copy of current if this is user's first re-embedding
                backup.op(P4.user_PrD, 1) #makes copy in Topos/PrD and DiffMaps
                P4.origEmbed[P4.user_PrD - 1] = 0
                np.savetxt(P4.origEmbedFile, P4.origEmbed, fmt='%i')
            
            Manifold2dCanvas.pts_orig, pts_orig_zip = itertools.tee(Manifold2dCanvas.pts_orig)
            Manifold2dCanvas.pts_new, pts_new_zip = itertools.tee(Manifold2dCanvas.pts_new)

            embedd.op(list(pts_orig_zip),list(pts_new_zip), P4.user_PrD - 1) #updates all manifold files for PD

            self.start_task1()

        else:
            pass

    def revert(self):
        msg = "Performing this action will revert the manifold for the \
                current PD back to its original embedding.\
                <br /><br />\
                Do you want to proceed?"
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s' % progname)
        box.setText('<b>Revert Manifold</b>')
        box.setFont(font_standard)
        box.setIcon(QtGui.QMessageBox.Question)
        box.setInformativeText(msg)
        box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
        reply = box.exec_()
        if reply == QtGui.QMessageBox.Yes:
            Manifold2dCanvas.btn_reset.setDisabled(False)
            self.btn_rebed.setDisabled(True)
            self.btn_revert.setDisabled(True)

            vid_tabs.setTabEnabled(0, False)
            vid_tabs.setTabEnabled(2, False)
            vid_tabs.setTabEnabled(3, False)
            vid_tabs.setTabEnabled(4, False)
            vid_tabs.setTabEnabled(5, False)

            P4.origEmbed[P4.user_PrD - 1] = 1
            backup.op(P4.user_PrD, -1)
            np.savetxt(P4.origEmbedFile, P4.origEmbed, fmt='%i')

            psi_file = '{}prD_{}'.format(p.psi_file, P4.user_PrD - 1) #current embedding
            data = myio.fin1(psi_file)
            x = data['psi'][:,Manifold2dCanvas.eigChoice1]
            y = data['psi'][:,Manifold2dCanvas.eigChoice2]

            # redraw and resize figure:
            self.ax.clear()
            Manifold2dCanvas.pts_orig = zip(x,y)
            Manifold2dCanvas.pts_origX = x
            Manifold2dCanvas.pts_origY = y
            for i in Manifold2dCanvas.pts_orig:
                x,y = i
                self.ax.scatter(x,y,s=1,c='#1f77b4') #plot initial data, C0

            for i in Manifold2dCanvas.pts_orig:
                x,y = i
                self.ax.scatter(x,y,s=1,c='#1f77b4') #plot initial data, C0
            for tick in self.ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(4)
            for tick in self.ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(4)
            self.ax.get_xaxis().set_ticks([])
            self.ax.get_yaxis().set_ticks([])
            self.ax.set_title('Place points on the plot to encircle deviant cluster(s)', fontsize=3.5)
            self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (Manifold2dCanvas.eigChoice1+1), fontsize=6)
            self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (Manifold2dCanvas.eigChoice2+1), fontsize=6)
            self.ax.autoscale()
            self.canvas.draw()

            vid_tabs.setTabEnabled(0, True)
            vid_tabs.setTabEnabled(2, True)
            vid_tabs.setTabEnabled(3, True)
            vid_tabs.setTabEnabled(4, True)
            vid_tabs.setTabEnabled(5, True)

            msg = 'The manifold for PD %s has been successfully reverted.' % (P4.user_PrD)
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Reversion' % progname)
            box.setText('<b>Revert Manifold</b>')
            box.setIcon(QtGui.QMessageBox.Warning)
            box.setFont(font_standard)
            box.setInformativeText(msg)
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setDefaultButton(QtGui.QMessageBox.Ok)
            ret = box.exec_()

            # force-update main GUI window (topos images)
            holdPrD = P4.user_PrD
            if P4.user_PrD != 1:
                P4.entry_PrD.setValue(1)
            else:
                P4.entry_PrD.setValue(2)
            P4.entry_PrD.setValue(holdPrD)

            PrD_window.close()
        else:
            pass

    def view(self): #view average of all images in encircled region
        Manifold2dCanvas.pts_encircled = []
        Manifold2dCanvas.pts_encircledX = []
        Manifold2dCanvas.pts_encircledY = []

        codes = []
        for i in range(len(self.coordsX)):
            if i == 0:
                codes.extend([pltPath.Path.MOVETO])
            elif i == len(self.coordsX)-1:
                codes.extend([pltPath.Path.CLOSEPOLY])
            else:
                codes.extend([pltPath.Path.LINETO])
        
        path = pltPath.Path(list(map(list, zip(self.coordsX, self.coordsY))), codes)
        inside = path.contains_points(np.dstack((Manifold2dCanvas.pts_origX,Manifold2dCanvas.pts_origY))[0].tolist(), radius=1e-9)

        idx_encircled = []
        index = 0
        for i in inside:
            index += 1
            if i == True:
                Manifold2dCanvas.pts_encircledX.append(Manifold2dCanvas.pts_origX[index-1])
                Manifold2dCanvas.pts_encircledY.append(Manifold2dCanvas.pts_origY[index-1])
                Manifold2dCanvas.pts_encircled = zip(Manifold2dCanvas.pts_encircledX, Manifold2dCanvas.pts_encircledY)
                idx_encircled.append(index-1)

        Manifold2dCanvas.imgAvg = clusterAvg.op(idx_encircled, P4.user_PrD - 1)
        self.ClusterAvg()


    def ClusterAvg(self):
        global ClusterAvgMain_window
        try:
            ClusterAvgMain_window.close()
        except:
            pass
        #self.setWindowModality(QtCore.Qt.ApplicationModal) #freezes out parent window
        ClusterAvgMain_window = ClusterAvgMain()
        ClusterAvgMain_window.setMinimumSize(10, 10)
        ClusterAvgMain_window.setWindowTitle('Projection Direction %s' % (P4.user_PrD))
        ClusterAvgMain_window.show()


    def onclick(self, event):
        if self.connected == 0:
            ix, iy = event.xdata, event.ydata
            if ix != None and iy != None:
                self.coordsX.append(float(ix))
                self.coordsY.append(float(iy))
                ax = self.figure.axes[0]
                ax.plot(event.xdata, event.ydata, color='#d62728', marker='+', zorder=2) #on top, C3
                if len(self.coordsX) > 1:
                    x0, y0 = self.coordsX[-2], self.coordsY[-2]
                    x1, y1 = self.coordsX[-1], self.coordsY[-1]
                    ax.plot([x0,x1],[y0,y1],
                            color='#7f7f7f', linestyle='solid', linewidth=.5, zorder=1) #C7
                self.canvas.draw()
            if len(self.coordsX) > 2:
                self.btn_connect.setDisabled(False)

    ##########
    # Task 1:
    @QtCore.Slot()
    def start_task1(self):
        set_params.op(0) #send new GUI data to parameters file

        task1 = threading.Thread(target=psiAnalysis.op,
                         args=(self.progress1Changed, ))
        task1.daemon = True
        task1.start()

    @QtCore.Slot(int)
    def on_progress1Changed(self, val):
        Manifold2dCanvas.progress1.setValue(val/2)
        if val/2 == 50:
            gc.collect()
            self.start_task2()

    ##########
    # Task 2:
    @QtCore.Slot()
    def start_task2(self):
        set_params.op(0) #send new GUI data to parameters file

        task2 = threading.Thread(target=NLSAmovie.op,
                         args=(self.progress2Changed, ))
        task2.daemon = True
        task2.start()

    @QtCore.Slot(int)
    def on_progress2Changed(self, val):
        Manifold2dCanvas.progress1.setValue(val/2 + 50)
        if (val/2 + 50) == 100:
            gc.collect()

            vid_tabs.setTabEnabled(0, True)
            vid_tabs.setTabEnabled(2, True)
            vid_tabs.setTabEnabled(3, True)
            vid_tabs.setTabEnabled(4, True)
            vid_tabs.setTabEnabled(5, True)

            msg = 'The manifold for PD %s has been successfully re-embedded.' % (P4.user_PrD)
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Re-embedding' % progname)
            box.setText('<b>Re-embed Manifold</b>')
            box.setIcon(QtGui.QMessageBox.Warning)
            box.setFont(font_standard)
            box.setInformativeText(msg)
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setDefaultButton(QtGui.QMessageBox.Ok)
            ret = box.exec_()

            # force-update main GUI window (topos images)
            holdPrD = P4.user_PrD
            if P4.user_PrD != 1:
                P4.entry_PrD.setValue(1)
            else:
                P4.entry_PrD.setValue(2)
            P4.entry_PrD.setValue(holdPrD)

            PrD_window.close()

# =============================================================================
# Plot image average within encircled region:
# =============================================================================

class ClusterAvgMain(QtGui.QMainWindow):
    def __init__(self):
        super(ClusterAvgMain, self).__init__()
        self.left = 10
        self.top = 10
        self.initUI()

    def initUI(self):
        centralwidget = QtGui.QWidget()
        self.setCentralWidget(centralwidget)
        canvas = ClusterAvgCanvas(self, width=2, height=2)
        toolbar = NavigationToolbar(canvas, self)
        vbl = QtGui.QVBoxLayout(centralwidget)
        vbl.addWidget(toolbar)
        vbl.addWidget(canvas)
        self.show()

class ClusterAvgCanvas(FigureCanvas):
    def __init__(self, parent=None, width=2, height=2, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.updateGeometry()
        self.plot()

    def plot(self):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.set_title('Image Average', fontsize=6)
        ax.imshow(Manifold2dCanvas.imgAvg, cmap='gray')
        ax.axis('off')
        self.draw()


# =============================================================================
# Plot gaussian kernel bandwidth (via fergusonE.py):
# =============================================================================

class BandwidthMain(QtGui.QMainWindow):
    def __init__(self):
        super(BandwidthMain, self).__init__()
        self.left = 10
        self.top = 10
        self.initUI()

        # Sub-Help Menu:
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction('&Kernel Bandwidth', self.guide_bandwidth)

    def guide_bandwidth(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>Gaussian Kernel Bandwidth</b>')
        box.setFont(font_standard)
        box.setInformativeText("<span style='font-weight:normal;'>\
                                A log-log plot of the sum of the elements of the pairwise similarity matrix \
                                as a function of the Gaussian kernel bandwidth.\
                                <br /><br />\
                                The linear region delineates\
                                the range of suitable epsilon values. Twice its slope provides an estimate\
                                of the effective dimensionality.\
                                </span>")
        box.setStandardButtons(QtGui.QMessageBox.Ok)
        ret = box.exec_()

    def initUI(self):
        centralwidget = QtGui.QWidget()
        self.setCentralWidget(centralwidget)
        plot = BandwidthCanvas(self, width=5, height=4)
        toolbar =  NavigationToolbar(plot, self)
        vbl = QtGui.QVBoxLayout(centralwidget)
        vbl.addWidget(toolbar)
        vbl.addWidget(plot)
        self.show()

class BandwidthCanvas(FigureCanvas):
    logEps = 0
    logSumWij = 0
    popt = 0
    R_squared = 0
    
    def __init__(self, parent=None, width=5, height=4, dpi=200):
        psi_file = '{}prD_{}'.format(p.psi_file, P4.user_PrD - 1) #current embedding
        data = myio.fin1(psi_file)
        BandwidthCanvas.logEps = data['logEps']
        BandwidthCanvas.logSumWij = data['logSumWij']
        BandwidthCanvas.popt = data['popt']
        BandwidthCanvas.R_squared = data['R_squared']
        
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.clear()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.updateGeometry()
        fig.set_tight_layout(True)
        self.plot()

    def plot(self):
        
        def fun(xx, aa0, aa1, aa2, aa3): #fit tanh()
            #aa3: y-value of tanh inflection point
            #aa2: y-value of apex (asymptote)
            #aa1: x-value of x-shift (inverse sign)
            #aa0: alters spread
            F = aa3 + aa2 * np.tanh(aa0 * xx + aa1)
            return F
        
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.scatter(BandwidthCanvas.logEps, BandwidthCanvas.logSumWij, s=1, c='C0', edgecolor='C0', zorder=.1, label='data')
        ax.plot(BandwidthCanvas.logEps, fun(BandwidthCanvas.logEps,
                                                 BandwidthCanvas.popt[0],
                                                 BandwidthCanvas.popt[1],
                                                 BandwidthCanvas.popt[2],
                                                 BandwidthCanvas.popt[3]), c='C1', linewidth=.5, zorder=.2, label=r'$\mathrm{tanh(x)}$')    
        ax.axvline(-(BandwidthCanvas.popt[1] / BandwidthCanvas.popt[0]), color='C2', linewidth=.5, linestyle='-', zorder=0, label=r'$\mathrm{ln \ \epsilon}$')
        ax.plot(BandwidthCanvas.logEps, BandwidthCanvas.popt[0]*BandwidthCanvas.popt[2]*(BandwidthCanvas.logEps+BandwidthCanvas.popt[1]/BandwidthCanvas.popt[0]) + BandwidthCanvas.popt[3],
                c='C3', linewidth=.5, zorder=.3, label='slope')
        ax.set_ylim(np.amin(fun(BandwidthCanvas.logEps, BandwidthCanvas.popt[0], BandwidthCanvas.popt[1], BandwidthCanvas.popt[2], BandwidthCanvas.popt[3]))-1, 
                    np.amax(fun(BandwidthCanvas.logEps, BandwidthCanvas.popt[0], BandwidthCanvas.popt[1], BandwidthCanvas.popt[2], BandwidthCanvas.popt[3]))+1)
        ax.legend(loc='lower right', fontsize=6)
        
        slope = BandwidthCanvas.popt[0]*BandwidthCanvas.popt[2] #slope of tanh
        
        textstr = '\n'.join((
                    r'$y=%.2f + %.2f tanh(%.2fx + %.2f)$' % (BandwidthCanvas.popt[3], BandwidthCanvas.popt[2], BandwidthCanvas.popt[0], BandwidthCanvas.popt[1]),
                    r'$\mathrm{Slope=%.2f}$' % (slope, ),
                    r'$\mathrm{Optimal \ log(\epsilon)=%.2f}$' % (-(BandwidthCanvas.popt[1] / BandwidthCanvas.popt[0]), ),
                    r'$\mathrm{R^2}=%.2f$' % (BandwidthCanvas.R_squared, ),))
                    
        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)

        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=6,
                verticalalignment='top', bbox=props)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        ax.set_title('Gaussian Kernel Bandwidth',fontsize=8)
        ax.set_xlabel(r'$\mathrm{ln \ \epsilon}$', fontsize=8)
        ax.set_ylabel(r'$\mathrm{ln \ \sum_{i,j} \ A_{i,j}}$', fontsize=8, rotation=90)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.offsetText.set_fontsize(6)
        ax.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)
        #ax.autoscale()
        self.draw()

# =============================================================================
# Plot eigenvalue spectrum:
# =============================================================================

class EigValMain(QtGui.QMainWindow):
    def __init__(self):
        super(EigValMain, self).__init__()
        self.left = 10
        self.top = 10
        self.initUI()

    def initUI(self):
        centralwidget = QtGui.QWidget()
        self.setCentralWidget(centralwidget)
        plot = EigValCanvas(self, width=5, height=4)
        toolbar =  NavigationToolbar(plot, self)
        vbl = QtGui.QVBoxLayout(centralwidget)
        vbl.addWidget(toolbar)
        vbl.addWidget(plot)
        self.show()

class EigValCanvas(FigureCanvas):
    # all eigenvecs/vals:
    eig_n = []
    eig_v = []
    # user-computed vecs/vals (color blue):
    eig_n1 = []
    eig_v1 = []
    # remaining vecs/vals via [eig_n - eig_n1] (color gray):
    eig_n2 = []
    eig_v2 = []

    def __init__(self, parent=None, width=5, height=4, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.clear()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.updateGeometry()
        fig.set_tight_layout(True)
        self.plot()

    def EigValRead(self):

        EigValCanvas.eig_n = []
        EigValCanvas.eig_v = []
        EigValCanvas.eig_n1 = []
        EigValCanvas.eig_v1 = []
        EigValCanvas.eig_n2 = []
        EigValCanvas.eig_v2 = []

        fname = os.path.join(P1.user_directory,'outputs_%s/topos/PrD_%s/eig_spec.txt' % (p.proj_name,P4.user_PrD))
        data = []
        with open(fname) as values:
            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                data.append(column)
        col1 = data[0]
        col2 = data[1]
        cols = np.column_stack((col1, col2))

        for i,j in cols:
            EigValCanvas.eig_n.append(int(i))
            EigValCanvas.eig_v.append(float(j))
            if int(i) <= int(p.num_psis):
                EigValCanvas.eig_n1.append(int(i))
                EigValCanvas.eig_v1.append(float(j))
            else:
                EigValCanvas.eig_n2.append(int(i))
                EigValCanvas.eig_v2.append(float(j))
        return

    def plot(self):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.bar(EigValCanvas.eig_n1, EigValCanvas.eig_v1, edgecolor='none', color='#1f77b4', align='center') #C0: blue
        ax.bar(EigValCanvas.eig_n2, EigValCanvas.eig_v2, edgecolor='none', color='#7f7f7f', align='center') #C7: gray

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        ax.set_title('Eigenvalue Spectrum',fontsize=8)
        ax.set_xlabel(r'$\mathrm{\Psi}$', fontsize=8)
        ax.set_ylabel(r'$\mathrm{\lambda}$', fontsize=8, rotation=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axhline(0, color='k', linestyle='-', linewidth=.25)
        ax.get_xaxis().set_tick_params(direction='out', width=.25, length=2)
        ax.get_yaxis().set_tick_params(direction='out', width=.25, length=2)
        ax.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(6)
        ax.set_xticks(EigValCanvas.eig_n)
        ax.autoscale()
        self.draw()

class Manifold3dCanvas(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Manifold3dCanvas, self).__init__(parent)

        self.PrD = P4.user_PrD
        self.initComplete = 0
        self.eigVals = []
        
        eig_fname = os.path.join(P1.user_directory,'outputs_%s/topos/PrD_%s/eig_spec.txt' % (p.proj_name, P4.user_PrD))
        data = []
        with open(eig_fname) as values:
            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                data.append(column)
        for i in data[1]:
            self.eigVals.append(float(i))

        # psi from diffusion maps:
        outDirDiff = os.path.join(P1.user_directory,'outputs_{}/diff_maps'.format(p.proj_name))
        diff_fname = open(os.path.join(outDirDiff, 'gC_trimmed_psi_prD_%s' % (self.PrD-1)), 'rb')
        diff_data = pickle.load(diff_fname)
        self.psi = diff_data['psi'] #* self.eigVals #scale eigenvectors by corresponding eigenvalues
        self.psi1 = self.psi[:, Manifold2dCanvas.eigChoice1]
        self.psi2 = self.psi[:, Manifold2dCanvas.eigChoice2]
        self.psi3 = self.psi[:, Manifold2dCanvas.eigChoice3]
        diff_fname.close()

        # create canvas and plot data:
        self.figure = Figure(dpi=200)
        Manifold3dCanvas.canvas = FigureCanvas(self.figure)
        #self.toolbar = NavigationToolbar(Manifold3dCanvas.canvas, self)
        self.ax = Axes3D(self.figure)
        self.ax.mouse_init()
        self.ax.view_init(90, -90)

        # Matplotlib Default Colors:
        # ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        self.ax.scatter(self.psi1, self.psi2, self.psi3, linewidths= .5, edgecolors='k', c='#d62728') #C3

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in self.ax.zaxis.get_major_ticks():
            tick.label.set_fontsize(4)

        self.ax.tick_params(axis='x', which='major', pad=-3)
        self.ax.tick_params(axis='y', which='major', pad=-3)
        self.ax.tick_params(axis='z', which='major', pad=-3)
        self.ax.xaxis.labelpad=-8
        self.ax.yaxis.labelpad=-8
        self.ax.zaxis.labelpad=-8

        self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (Manifold2dCanvas.eigChoice1+1), fontsize=6)
        self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (Manifold2dCanvas.eigChoice2+1), fontsize=6)
        self.ax.set_zlabel(r'$\mathrm{\Psi}$%s' % (Manifold2dCanvas.eigChoice3+1), fontsize=6)

        Manifold3dCanvas.canvas.draw() #refresh canvas

        # canvas buttons:
        self.label_X = QtGui.QLabel('X-axis:')
        self.label_X.setFont(font_standard)
        self.label_X.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.label_Y = QtGui.QLabel('Y-axis:')
        self.label_Y.setFont(font_standard)
        self.label_Y.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.label_Z = QtGui.QLabel('Z-axis:')
        self.label_Z.setFont(font_standard)
        self.label_Z.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.combo_X = QtGui.QComboBox(self)
        self.combo_X.setFont(font_standard)
        self.combo_X.currentIndexChanged.connect(self.choose_X)
        self.combo_X.setCurrentIndex(Manifold2dCanvas.eigChoice1)
        self.combo_X.setDisabled(False)

        self.combo_Y = QtGui.QComboBox(self)
        self.combo_Y.setFont(font_standard)
        self.combo_Y.currentIndexChanged.connect(self.choose_Y)
        self.combo_Y.setCurrentIndex(Manifold2dCanvas.eigChoice2)
        self.combo_Y.setDisabled(False)

        self.combo_Z = QtGui.QComboBox(self)
        self.combo_Z.setFont(font_standard)
        self.combo_Z.currentIndexChanged.connect(self.choose_Z)
        self.combo_Z.setCurrentIndex(Manifold2dCanvas.eigChoice3)
        self.combo_Z.setDisabled(False)

        for psi in range(p.num_psiTrunc):
            self.combo_X.addItem('Psi %s' % (int(psi+1)))
            self.combo_Y.addItem('Psi %s' % (int(psi+1)))
            self.combo_Z.addItem('Psi %s' % (int(psi+1)))

        self.combo_X.setCurrentIndex(Manifold2dCanvas.eigChoice1)
        self.combo_Y.setCurrentIndex(Manifold2dCanvas.eigChoice2)
        self.combo_Z.setCurrentIndex(Manifold2dCanvas.eigChoice3)
        self.combo_X.model().item(Manifold2dCanvas.eigChoice2).setEnabled(False)
        self.combo_X.model().item(Manifold2dCanvas.eigChoice3).setEnabled(False)
        self.combo_Y.model().item(Manifold2dCanvas.eigChoice1).setEnabled(False)
        self.combo_Y.model().item(Manifold2dCanvas.eigChoice3).setEnabled(False)
        self.combo_Z.model().item(Manifold2dCanvas.eigChoice1).setEnabled(False)
        self.combo_Z.model().item(Manifold2dCanvas.eigChoice2).setEnabled(False)

        self.initComplete = 1

        layout = QtGui.QGridLayout()
        layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        #layout.addWidget(self.toolbar, 0,0,1,4)
        layout.addWidget(Manifold3dCanvas.canvas, 1,0,4,7)

        layout.addWidget(self.label_X, 5,0,1,1)
        layout.addWidget(self.label_Y, 5,2,1,1)
        layout.addWidget(self.label_Z, 5,4,1,1)
        layout.addWidget(self.combo_X, 5,1,1,1)
        layout.addWidget(self.combo_Y, 5,3,1,1)
        layout.addWidget(self.combo_Z, 5,5,1,1)

        self.setLayout(layout)

    def replot(self):
        if self.initComplete == 1:
            # redraw and resize figure:
            self.ax.clear()
            self.ax.scatter(self.psi1, self.psi2, self.psi3, label='psi_dif', linewidths= .5, edgecolors='k', c='#d62728') #C3

            for tick in self.ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(4)
            for tick in self.ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(4)
            for tick in self.ax.zaxis.get_major_ticks():
                tick.label.set_fontsize(4)

            self.ax.tick_params(axis='x', which='major', pad=-3)
            self.ax.tick_params(axis='y', which='major', pad=-3)
            self.ax.tick_params(axis='z', which='major', pad=-3)
            self.ax.xaxis.labelpad=-8
            self.ax.yaxis.labelpad=-8
            self.ax.zaxis.labelpad=-8
            self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (int(self.combo_X.currentIndex())+1), fontsize=6)
            self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (int(self.combo_Y.currentIndex())+1), fontsize=6)
            self.ax.set_zlabel(r'$\mathrm{\Psi}$%s' % (int(self.combo_Z.currentIndex())+1), fontsize=6)

            Manifold3dCanvas.canvas.draw()

    def choose_X(self):
        if self.initComplete == 1:
            x = int(self.combo_X.currentIndex())
            self.psi1 = self.psi[:,x]
            
            # update choice on the 2D Embedding tab:
            Manifold2dCanvas.eigChoice1 = x
            psi_file = '{}prD_{}'.format(p.psi_file, P4.user_PrD - 1) #current embedding
            data = myio.fin1(psi_file)
            x = data['psi'][:,Manifold2dCanvas.eigChoice1]
            y = data['psi'][:,Manifold2dCanvas.eigChoice2]
            Manifold2dCanvas.pts_orig = zip(x,y)
            Manifold2dCanvas.pts_origX = x
            Manifold2dCanvas.pts_origY = y
            Manifold2dCanvas.btn_reset.click()
    
            if self.initComplete == 1:
    
                for i in range(p.num_psiTrunc):
                    self.combo_Y.model().item(i).setEnabled(True)
                    self.combo_Z.model().item(i).setEnabled(True)
    
                self.combo_Y.model().item(int(self.combo_X.currentIndex())).setEnabled(False)
                self.combo_Y.model().item(int(self.combo_Z.currentIndex())).setEnabled(False)
                self.combo_Z.model().item(int(self.combo_X.currentIndex())).setEnabled(False)
                self.combo_Z.model().item(int(self.combo_Y.currentIndex())).setEnabled(False)
    
            self.replot()

    def choose_Y(self):
        if self.initComplete == 1:
            y = int(self.combo_Y.currentIndex())
            self.psi2 = self.psi[:,y]
            
            # update choice on the 2D Embedding tab:
            Manifold2dCanvas.eigChoice2 = y
            psi_file = '{}prD_{}'.format(p.psi_file, P4.user_PrD - 1) #current embedding
            data = myio.fin1(psi_file)
            x = data['psi'][:,Manifold2dCanvas.eigChoice1]
            y = data['psi'][:,Manifold2dCanvas.eigChoice2]
            Manifold2dCanvas.pts_orig = zip(x,y)
            Manifold2dCanvas.pts_origX = x
            Manifold2dCanvas.pts_origY = y
            Manifold2dCanvas.btn_reset.click()
    
            if self.initComplete == 1:
    
                for i in range(p.num_psiTrunc):
                    self.combo_X.model().item(i).setEnabled(True)
                    self.combo_Z.model().item(i).setEnabled(True)
    
                self.combo_X.model().item(int(self.combo_Y.currentIndex())).setEnabled(False)
                self.combo_X.model().item(int(self.combo_Z.currentIndex())).setEnabled(False)
                self.combo_Z.model().item(int(self.combo_X.currentIndex())).setEnabled(False)
                self.combo_Z.model().item(int(self.combo_Y.currentIndex())).setEnabled(False)
    
            self.replot()

    def choose_Z(self):
        if self.initComplete == 1:
            z = int(self.combo_Z.currentIndex())
            self.psi3 = self.psi[:,z]
            
            Manifold2dCanvas.eigChoice3 = z
    
            if self.initComplete == 1:
    
                for i in range(p.num_psiTrunc):
                    self.combo_X.model().item(i).setEnabled(True)
                    self.combo_Y.model().item(i).setEnabled(True)
    
                self.combo_X.model().item(int(self.combo_Y.currentIndex())).setEnabled(False)
                self.combo_X.model().item(int(self.combo_Z.currentIndex())).setEnabled(False)
                self.combo_Y.model().item(int(self.combo_X.currentIndex())).setEnabled(False)
                self.combo_Y.model().item(int(self.combo_Z.currentIndex())).setEnabled(False)
    
            self.replot()


class PsiCanvas(QtGui.QDialog):
    def __init__(self, parent=None):
        super(PsiCanvas, self).__init__(parent)

        self.PrD = P4.user_PrD
        self.con_on = 0
        self.rec_on = 1
        self.initComplete = 0

        # psis from psi analsis:
        self.outDirPsi = os.path.join(P1.user_directory,'outputs_{}/psi_analysis'.format(p.proj_name))
        self.psi_fname = open(os.path.join(self.outDirPsi, 'S2_prD_%s_psi_%s' % (self.PrD-1, Manifold2dCanvas.eig_current-1)), 'rb')
        self.psi_data = pickle.load(self.psi_fname)
        #PsiC:
        self.psiC = self.psi_data['psiC1']
        self.psiC1 = self.psiC[:,0]
        self.psiC2 = self.psiC[:,1]
        self.psiC3 = self.psiC[:,2]
        #Psirec 1:
        self.psirec = self.psi_data['psirec']
        self.psirec1 = self.psirec[:,0]
        self.psirec2 = self.psirec[:,1]
        self.psirec3 = self.psirec[:,2]
        self.psi_fname.close()

        # create canvas and plot data:
        self.figure = Figure(dpi=200)
        self.canvas = FigureCanvas(self.figure)
        #self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = Axes3D(self.figure)
        self.ax.mouse_init()

        self.ax.view_init(90, 90)

        # Matplotlib Default Colors:
        # ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        self.ax.scatter(self.psirec1, self.psirec2, self.psirec3, label='psi_rec', linewidths=.5, edgecolors='k', color='#1f77b4') #C0
        #self.ax.scatter(self.psiC1, self.psiC2, self.psiC3, label='psi_con', linewidths=.5, edgecolors='k', c='#2ca02c') #C2
        self.ax.legend(loc='best', prop={'size': 6})

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in self.ax.zaxis.get_major_ticks():
            tick.label.set_fontsize(4)

        self.ax.tick_params(axis='x', which='major', pad=-3)
        self.ax.tick_params(axis='y', which='major', pad=-3)
        self.ax.tick_params(axis='z', which='major', pad=-3)
        self.ax.xaxis.labelpad=-8
        self.ax.yaxis.labelpad=-8
        self.ax.zaxis.labelpad=-8

        self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (1), fontsize=6)
        self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (2), fontsize=6)
        self.ax.set_zlabel(r'$\mathrm{\Psi}$%s' % (3), fontsize=6)

        self.canvas.draw() #refresh canvas

        # canvas buttons:
        self.label_X = QtGui.QLabel('X-axis:')
        self.label_X.setFont(font_standard)
        self.label_X.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.label_Y = QtGui.QLabel('Y-axis:')
        self.label_Y.setFont(font_standard)
        self.label_Y.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.label_Z = QtGui.QLabel('Z-axis:')
        self.label_Z.setFont(font_standard)
        self.label_Z.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.combo_X = QtGui.QComboBox(self)
        self.combo_X.setFont(font_standard)
        self.combo_X.currentIndexChanged.connect(self.choose_X)
        self.combo_X.setDisabled(False)

        self.combo_Y = QtGui.QComboBox(self)
        self.combo_Y.setFont(font_standard)
        self.combo_Y.currentIndexChanged.connect(self.choose_Y)
        self.combo_Y.setDisabled(False)

        self.combo_Z = QtGui.QComboBox(self)
        self.combo_Z.setFont(font_standard)
        self.combo_Z.currentIndexChanged.connect(self.choose_Z)
        self.combo_Z.setDisabled(False)

        for psi in range(p.num_psiTrunc):
            self.combo_X.addItem('Psi %s' % (int(psi+1)))
            self.combo_Y.addItem('Psi %s' % (int(psi+1)))
            self.combo_Z.addItem('Psi %s' % (int(psi+1)))
        self.combo_X.setCurrentIndex(0)
        self.combo_Y.setCurrentIndex(1)
        self.combo_Z.setCurrentIndex(2)
        self.combo_X.model().item(1).setEnabled(False)
        self.combo_X.model().item(2).setEnabled(False)
        self.combo_Y.model().item(0).setEnabled(False)
        self.combo_Y.model().item(2).setEnabled(False)
        self.combo_Z.model().item(0).setEnabled(False)
        self.combo_Z.model().item(1).setEnabled(False)

        self.initComplete = 1

        self.label_Vline = QtGui.QLabel('') #separating line
        self.label_Vline.setFont(font_standard)
        self.label_Vline.setFrameStyle(QtGui.QFrame.VLine | QtGui.QFrame.Sunken)

        self.check_psiC = QtGui.QCheckBox('Concatenation')
        self.check_psiC.clicked.connect(self.choose_con)
        self.check_psiC.setChecked(False)

        self.check_psiR = QtGui.QCheckBox('Reconstruction')
        self.check_psiR.clicked.connect(self.choose_rec)
        self.check_psiR.setChecked(True)

        layout = QtGui.QGridLayout()
        layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        #layout.addWidget(self.toolbar, 0,0,1,4)
        layout.addWidget(self.canvas, 1,0,4,4)

        layout.addWidget(self.label_X, 5,0,1,1)
        layout.addWidget(self.label_Y, 6,0,1,1)
        layout.addWidget(self.label_Z, 7,0,1,1)
        layout.addWidget(self.combo_X, 5,1,1,1)
        layout.addWidget(self.combo_Y, 6,1,1,1)
        layout.addWidget(self.combo_Z, 7,1,1,1)
        layout.addWidget(self.label_Vline, 5,2,3,1)
        layout.addWidget(self.check_psiC, 5,3,1,1)
        layout.addWidget(self.check_psiR, 6,3,1,1)

        self.setLayout(layout)

    def replot(self):
        if self.initComplete == 1:
            # redraw and resize figure:
            self.ax.clear()

            if self.rec_on == 1:
                self.ax.scatter(self.psirec1, self.psirec2, self.psirec3, label='psi_rec', linewidths=.5, edgecolors='k', color='#1f77b4') #C0
            if self.con_on == 1:
                self.ax.scatter(self.psiC1, self.psiC2, self.psiC3, label='psi_con', linewidths=.5, edgecolors='k', c='#2ca02c') #C2

            if self.rec_on == 1 or self.con_on == 1:
                self.ax.legend(loc='best', prop={'size': 6})

            for tick in self.ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(4)
            for tick in self.ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(4)
            for tick in self.ax.zaxis.get_major_ticks():
                tick.label.set_fontsize(4)

            self.ax.tick_params(axis='x', which='major', pad=-3)
            self.ax.tick_params(axis='y', which='major', pad=-3)
            self.ax.tick_params(axis='z', which='major', pad=-3)
            self.ax.xaxis.labelpad=-8
            self.ax.yaxis.labelpad=-8
            self.ax.zaxis.labelpad=-8
            self.ax.set_xlabel(r'$\mathrm{\Psi}$%s' % (int(self.combo_X.currentIndex())+1), fontsize=6)
            self.ax.set_ylabel(r'$\mathrm{\Psi}$%s' % (int(self.combo_Y.currentIndex())+1), fontsize=6)
            self.ax.set_zlabel(r'$\mathrm{\Psi}$%s' % (int(self.combo_Z.currentIndex())+1), fontsize=6)

            self.canvas.draw()

    def choose_X(self):
        x = int(self.combo_X.currentIndex())

        self.psiC1 = self.psiC[:,x]
        self.psirec1 = self.psirec[:,x]

        if self.initComplete == 1:

            for i in range(p.num_psiTrunc):
                self.combo_Y.model().item(i).setEnabled(True)
                self.combo_Z.model().item(i).setEnabled(True)

            self.combo_Y.model().item(int(self.combo_X.currentIndex())).setEnabled(False)
            self.combo_Y.model().item(int(self.combo_Z.currentIndex())).setEnabled(False)
            self.combo_Z.model().item(int(self.combo_X.currentIndex())).setEnabled(False)
            self.combo_Z.model().item(int(self.combo_Y.currentIndex())).setEnabled(False)

        self.replot()

    def choose_Y(self):
        y = int(self.combo_Y.currentIndex())

        self.psiC2 = self.psiC[:,y]
        self.psirec2 = self.psirec[:,y]

        if self.initComplete == 1:

            for i in range(p.num_psiTrunc):
                self.combo_X.model().item(i).setEnabled(True)
                self.combo_Z.model().item(i).setEnabled(True)

            self.combo_X.model().item(int(self.combo_Y.currentIndex())).setEnabled(False)
            self.combo_X.model().item(int(self.combo_Z.currentIndex())).setEnabled(False)
            self.combo_Z.model().item(int(self.combo_X.currentIndex())).setEnabled(False)
            self.combo_Z.model().item(int(self.combo_Y.currentIndex())).setEnabled(False)

        self.replot()

    def choose_Z(self):
        z = int(self.combo_Z.currentIndex())

        self.psiC3 = self.psiC[:,z]
        self.psirec3 = self.psirec[:,z]

        if self.initComplete == 1:

            for i in range(p.num_psiTrunc):
                self.combo_X.model().item(i).setEnabled(True)
                self.combo_Y.model().item(i).setEnabled(True)

            self.combo_X.model().item(int(self.combo_Y.currentIndex())).setEnabled(False)
            self.combo_X.model().item(int(self.combo_Z.currentIndex())).setEnabled(False)
            self.combo_Y.model().item(int(self.combo_X.currentIndex())).setEnabled(False)
            self.combo_Y.model().item(int(self.combo_Z.currentIndex())).setEnabled(False)

        self.replot()

    def choose_con(self):
        if self.check_psiC.isChecked():
            self.con_on = 1
        else:
            self.con_on = 0
        self.replot()

    def choose_rec(self):
        if self.check_psiR.isChecked():
            self.rec_on = 1
        else:
            self.rec_on = 0
        self.replot()


class ChronosCanvas(QtGui.QDialog):
    def __init__(self, parent=None):
        super(ChronosCanvas, self).__init__(parent)

        self.PrD = P4.user_PrD

        # chronos from psi analsis:
        self.outDirChr = os.path.join(P1.user_directory,'outputs_{}/psi_analysis'.format(p.proj_name))
        self.chr_fname = open(os.path.join(self.outDirChr, 'S2_prD_%s_psi_%s' % (self.PrD-1, Manifold2dCanvas.eig_current-1)), 'rb')
        self.chr_data = pickle.load(self.chr_fname)

        chronos = self.chr_data['VX']
        chronos1 = chronos[0]
        chronos2 = chronos[1]
        chronos3 = chronos[2]
        chronos4 = chronos[3]
        chronos5 = chronos[4]
        chronos6 = chronos[5]
        chronos7 = chronos[6]
        chronos8 = chronos[7]
        self.chr_fname.close()

        # create canvas and plot data:
        self.figure = Figure(dpi=200)
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        #self.toolbar = NavigationToolbar(self.canvas, self)

        fst = 6 #title font size
        lft = -25 #lower xlim

        self.ax1 = self.figure.add_subplot(2,4,1)
        self.ax1.plot(chronos1, color="#1f77b4", linewidth=.5)
        self.ax1.set_title('Chronos 1', fontsize=fst)
        self.ax1.set_xlim(left=lft, right=len(chronos1)-lft)
        self.ax1.set_xticks(np.arange(0, len(chronos1)+1, len(chronos1)/2))
        self.ax1.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)

        self.ax2 = self.figure.add_subplot(2,4,2, sharey=self.ax1)
        self.ax2.plot(chronos2, color="#1f77b4", linewidth=.5)
        self.ax2.set_title('Chronos 2', fontsize=fst)
        self.ax2.set_xlim(left=lft, right=len(chronos2)-lft)
        self.ax2.set_xticks(np.arange(0, len(chronos1)+1, len(chronos1)/2))
        self.ax2.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)

        self.ax3 = self.figure.add_subplot(2,4,3, sharey=self.ax1)
        self.ax3.plot(chronos3, color="#1f77b4", linewidth=.5)
        self.ax3.set_title('Chronos 3', fontsize=fst)
        self.ax3.set_xlim(left=lft, right=len(chronos3)-lft)
        self.ax3.set_xticks(np.arange(0, len(chronos1)+1, len(chronos1)/2))
        self.ax3.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)

        self.ax4 = self.figure.add_subplot(2,4,4, sharey=self.ax1)
        self.ax4.plot(chronos4, color="#1f77b4", linewidth=.5)
        self.ax4.set_title('Chronos 4', fontsize=fst)
        self.ax4.set_xlim(left=lft, right=len(chronos4)-lft)
        self.ax4.set_xticks(np.arange(0, len(chronos1)+1, len(chronos1)/2))
        self.ax4.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)

        self.ax5 = self.figure.add_subplot(2,4,5, sharey=self.ax1)
        self.ax5.plot(chronos5, color="#1f77b4", linewidth=.5)
        self.ax5.set_title('Chronos 5', fontsize=fst)
        self.ax5.set_xlim(left=lft, right=len(chronos5)-lft)
        self.ax5.set_xticks(np.arange(0, len(chronos1)+1, len(chronos1)/2))
        self.ax5.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)

        self.ax6 = self.figure.add_subplot(2,4,6, sharey=self.ax1)
        self.ax6.plot(chronos6, color="#1f77b4", linewidth=.5)
        self.ax6.set_title('Chronos 6', fontsize=fst)
        self.ax6.set_xlim(left=lft, right=len(chronos6)-lft)
        self.ax6.set_xticks(np.arange(0, len(chronos1)+1, len(chronos1)/2))
        self.ax6.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)

        self.ax7 = self.figure.add_subplot(2,4,7, sharey=self.ax1)
        self.ax7.plot(chronos7, color="#1f77b4", linewidth=.5)
        self.ax7.set_title('Chronos 7', fontsize=fst)
        self.ax7.set_xlim(left=lft, right=len(chronos7)-lft)
        self.ax7.set_xticks(np.arange(0, len(chronos1)+1, len(chronos1)/2))
        self.ax7.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)

        self.ax8 = self.figure.add_subplot(2,4,8, sharey=self.ax1)
        self.ax8.plot(chronos8, color="#1f77b4", linewidth=.5)
        self.ax8.set_title('Chronos 8', fontsize=fst)
        self.ax8.set_xlim(left=lft, right=len(chronos8)-lft)
        self.ax8.set_xticks(np.arange(0, len(chronos1)+1, len(chronos1)/2))
        self.ax8.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)

        fsa = 4 #axis font size
        for tick in self.ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax2.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax3.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax4.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax5.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax6.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax7.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax8.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax3.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax4.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax5.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax6.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax7.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)
        for tick in self.ax8.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsa)

        self.ax1.tick_params(direction='in', length=2, width=.25)
        self.ax2.tick_params(direction='in', length=2, width=.25)
        self.ax3.tick_params(direction='in', length=2, width=.25)
        self.ax4.tick_params(direction='in', length=2, width=.25)
        self.ax5.tick_params(direction='in', length=2, width=.25)
        self.ax6.tick_params(direction='in', length=2, width=.25)
        self.ax7.tick_params(direction='in', length=2, width=.25)
        self.ax8.tick_params(direction='in', length=2, width=.25)

        sides = ['top','bottom','left','right']
        for i in sides:
            self.ax1.spines[i].set_linewidth(1)
            self.ax2.spines[i].set_linewidth(1)
            self.ax3.spines[i].set_linewidth(1)
            self.ax4.spines[i].set_linewidth(1)
            self.ax5.spines[i].set_linewidth(1)
            self.ax6.spines[i].set_linewidth(1)
            self.ax7.spines[i].set_linewidth(1)
            self.ax8.spines[i].set_linewidth(1)

        self.canvas.draw() #refresh canvas

        layout = QtGui.QGridLayout()
        layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        #layout.addWidget(self.toolbar, 0,0,1,4)
        layout.addWidget(self.canvas, 1,0,4,4)

        self.setLayout(layout)


class TauCanvas(QtGui.QDialog):
    def __init__(self, parent=None):
        super(TauCanvas, self).__init__(parent)

        PrD = P4.user_PrD
        
        # tau from psi analsis:
        outDirTau = os.path.join(P1.user_directory,'outputs_{}/psi_analysis'.format(p.proj_name))
        tau_fname = open(os.path.join(outDirTau, 'S2_prD_%s_psi_%s' % (PrD-1, Manifold2dCanvas.eig_current-1)), 'rb')
        tau_data = pickle.load(tau_fname)
        tau = tau_data['tau']
        tau_fname.close()
        taus_val = []
        taus_num = []

        # create canvas and plot data:
        self.figure = Figure(dpi=200)
        self.figure.set_tight_layout(True)

        self.canvas = FigureCanvas(self.figure)
        #self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax1 = self.figure.add_subplot(1,2,1)
        self.ax2 = self.figure.add_subplot(1,2,2)

        idx = 0
        for i in tau:
            taus_val.append(i)
            taus_num.append(idx)
            idx += 1

        self.ax1.scatter(taus_val, taus_num, linewidths=.1, s=1,
                         edgecolors='k', c=taus_num, cmap='jet')
        self.ax2.hist(tau, bins=p.nClass, color='#1f77b4') #C0

        for tick in self.ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in self.ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in self.ax2.xaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in self.ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(4)
            
        #self.ax1.set_title('Parameterization', fontsize=6)
        self.ax1.set_xlabel('NLSA States', fontsize=5)
        #self.ax1.set_xlabel(r'$\mathrm{\tau}$', fontsize=7)
        self.ax1.set_ylabel('NLSA Image Indices', fontsize=5)

        self.ax1.set_xlim(xmin=0, xmax=1)
        self.ax1.set_ylim(ymin=0, ymax=np.shape(tau)[0])

        #self.ax2.set_title('PD Occupancy', fontsize=6)
        self.ax2.set_xlabel('NLSA States', fontsize=5)
        self.ax2.set_ylabel('NLSA Occupancy', fontsize=5)

        self.ax1.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)
        self.ax2.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)

        self.canvas.draw() #refresh canvas

        layout = QtGui.QGridLayout()
        layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        #layout.addWidget(self.toolbar, 0,0,1,4)
        layout.addWidget(self.canvas, 1,0,4,4)

        self.setLayout(layout)


################################################################################
# window for PrD occupancy thresholding:

class Thresh_Viz(QtGui.QMainWindow):
    def __init__(self):
        super(Thresh_Viz, self).__init__()
        self.left = 10
        self.top = 10
        self.initUI()

        # Sub-Help Menu:
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction('&PD Thresholding', self.guide_threshold)

    def guide_threshold(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>PD Thresholding</b>')
        box.setFont(font_standard)
        box.setInformativeText("<span style='font-weight:normal;'>\
                                The bar chart shows the number of particles within each\
                                projection direction (PD).\
                                <br /><br />\
                                The angular size of each PD is determined by the parameters set\
                                on the first tab. These parameters can be changed before\
                                proceeding to the next section <i>Embedding</i> tab) to alter the distribution\
                                of particles shown in the S2 plot.\
                                <br /><br />\
                                If a PD has fewer particles\
                                than the number specified by the low threshold parameter,\
                                that PD will be ignored across all future computations.\
                                If a PD has more particles\
                                than the number (<i>n</i>) specified by the high threshold parameter,\
                                only the first <i>n</i> particles under this threshold will be used for that PD.\
                                </span>")
        box.setStandardButtons(QtGui.QMessageBox.Ok)
        ret = box.exec_()

    def initUI(self):
        thresh_tab1 = ThreshAllCanvas(self)
        thresh_tab2 = ThreshFinalCanvas(self)
        thresh_tab3 = OccHistCanvas(self)
        global thresh_tabs
        thresh_tabs = QtGui.QTabWidget(self)
        thresh_tabs.addTab(thresh_tab1, 'Edit Thresholds')
        thresh_tabs.addTab(thresh_tab2, 'Thresholded PDs')
        thresh_tabs.addTab(thresh_tab3, 'Occupancy Distribution')

        style = """QTabWidget::tab-bar{
                alignment: center;
                }"""
        self.setStyleSheet(style)
        self.setCentralWidget(thresh_tabs)
        self.setWindowModality(QtCore.Qt.ApplicationModal) #freezes out parent window
        self.show()

        ThreshAllCanvas.thresh_low = p.PDsizeThL
        ThreshAllCanvas.thresh_high = p.PDsizeThH

        thresh_tabs.currentChanged.connect(self.onTabChange) #signal for tab changed via direct click

    def onTabChange(self, i):
        if i == 1: #signals when view switched to tab 2
            # re-threshold bins:
            temp_PrDs = []
            temp_occ = []
            temp_phi = []
            temp_theta = []
                        
            for i in range(0, len(P1.all_PrDs)):
                if P1.all_occ[i] >= ThreshAllCanvas.thresh_low:
                    temp_PrDs.append(i+1)
                    temp_occ.append(P1.all_occ[i])
                    # subtract 180 is needed for scatter's label switch:
                    temp_phi.append((float(P1.all_phi[i])-180)*np.pi/180) #needed in Radians
                    temp_theta.append(float(P1.all_theta[i]))

            #def format_coord(x,y):
                #return 'Phi={:1.2f}, Theta={:1.2f}'.format(((x*180)/np.pi)-180,y)

            # replot ThreshFinalCanvas:
            ThreshFinalCanvas.axes.clear()
            #ThreshFinalCanvas.axes.format_coord = format_coord
            ThreshFinalCanvas.cbar.remove()
            # thetas = [0,45,90,135,180,225,270,315] #in same order as labels below (ref only)
            theta_labels = ['%s180%s'%(u"\u00B1",u"\u00b0"),'-135%s'%(u"\u00b0"),'-90%s'%(u"\u00b0"),'-45%s'%(u"\u00b0"),
                            '0%s'%(u"\u00b0"),'45%s'%(u"\u00b0"),'90%s'%(u"\u00b0"),'135%s'%(u"\u00b0")]
            ThreshFinalCanvas.axes.set_ylim(0,180)
            ThreshFinalCanvas.axes.set_yticks(np.arange(0,180,20))
            ThreshFinalCanvas.axes.set_xticklabels(theta_labels)
            for tick in ThreshFinalCanvas.axes.xaxis.get_major_ticks():
                tick.label.set_fontsize(6)
            for tick in ThreshFinalCanvas.axes.yaxis.get_major_ticks():
                tick.label.set_fontsize(6)
            ThreshFinalCanvas.axes.grid(alpha=0.2)
            thresh = ThreshFinalCanvas.axes.scatter(temp_phi, temp_theta, edgecolor='k', linewidth=.5, c=temp_occ, s=5, cmap=cm.jet, vmin=0., vmax=float(np.amax(P1.all_occ)))
            thresh.set_alpha(0.75)
            ThreshFinalCanvas.cbar = ThreshFinalCanvas.figure.colorbar(thresh, pad=0.13)
            ThreshFinalCanvas.cbar.ax.tick_params(labelsize=6)
            ThreshFinalCanvas.cbar.ax.set_title(label='Occupancy',size=6)
            ThreshFinalCanvas.canvas.draw()

        if i == 2:
            OccHistCanvas.numBins = int(OccHistCanvas.entry_bins.value())
            # re-threshold bins:
            temp_PrDs = []
            temp_occ = []
            for i in range(0, len(P1.all_PrDs)):
                if P1.all_occ[i] >= ThreshAllCanvas.thresh_low:
                    temp_PrDs.append(i+1)
                    if P1.all_occ[i] > p.PDsizeThH:
                        temp_occ.append(p.PDsizeThH)
                    else:
                        temp_occ.append(P1.all_occ[i])

            OccHistCanvas.entry_bins.setValue(int(len(set(temp_occ))/2.))
            OccHistCanvas.entry_bins.setMaximum(len(set(temp_occ)))
            OccHistCanvas.entry_bins.setSuffix(' / %s' % len(set(temp_occ))) #number of unique occupancies

            # replot OccHistCanvas:
            OccHistCanvas.axes.clear()
            counts, bins, bars = OccHistCanvas.axes.hist(temp_occ, bins=int(OccHistCanvas.numBins), align='right',\
                                                            edgecolor='w', linewidth=1, color='#1f77b4') #C0

            OccHistCanvas.axes.set_xticks(bins)
            OccHistCanvas.axes.set_title('PD Occupancy Distribution', fontsize=6)
            OccHistCanvas.axes.set_xlabel('PD Occupancy', fontsize=5)
            OccHistCanvas.axes.set_ylabel('Number of PDs', fontsize=5)
            OccHistCanvas.axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            OccHistCanvas.axes.get_xaxis().get_major_formatter().set_scientific(False)
            OccHistCanvas.axes.ticklabel_format(useOffset=False, style='plain')

            for tick in OccHistCanvas.axes.xaxis.get_major_ticks():
                tick.label.set_fontsize(4)
            for tick in OccHistCanvas.axes.yaxis.get_major_ticks():
                tick.label.set_fontsize(4)

            OccHistCanvas.axes.autoscale()
            OccHistCanvas.figure.canvas.draw()


    def closeEvent(self, ce): #safety message if user clicks to exit via window button
        if ThreshAllCanvas.confirmed == 0:
            msg = 'Changes to the thresholding parameters have not been confirmed\
                    on the Edit Thresholds tab (via <i>Update Thresholds</i>).\
                    <br /><br />\
                    Do you want to proceed without saving?'
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Warning' % progname)
            box.setText('<b>Exit Warning</b>')
            box.setFont(font_standard)
            box.setIcon(QtGui.QMessageBox.Warning)
            box.setInformativeText(msg)
            box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
            reply = box.exec_()
            if reply == QtGui.QMessageBox.Yes:
                ThreshAllCanvas.confirmed = 1
                ThreshAllCanvas.thresh_low = p.PDsizeThL
                ThreshAllCanvas.thresh_high = p.PDsizeThH
                self.close()
            else:
                ce.ignore()
        if int(0) < ThreshAllCanvas.progBar.value() < int(100): #no escaping mid-thread
            ce.ignore()

class ThreshAllCanvas(QtGui.QDialog):
    # PrDs inside (in) and outside (out) current threshold window:
    thresh_low = p.PDsizeThL
    thresh_high = p.PDsizeThH
    in_occ = []
    in_PrDs = []
    out_occ = []
    out_PrDs = []
    xlimLo = 0
    xlimHi = 1
    confirmed = 1 #0:changes, 1:confirmed

    def __init__(self, parent=None):
        super(ThreshAllCanvas, self).__init__(parent)
        self.left = 10
        self.top = 10

        ThreshAllCanvas.progBar = QtGui.QProgressBar(self)#minimum=0,maximum=1,value=0)
        ThreshAllCanvas.progBar.setRange(0,100)
        ThreshAllCanvas.progBar.setValue(0)
        self.DataTask = DataThread()
        self.DataTask.DataFinished.connect(self.DataFinished)

        # create canvas and plot data:
        ThreshAllCanvas.figure = Figure(dpi=200)
        ThreshAllCanvas.figure.set_tight_layout(True)
        ThreshAllCanvas.canvas = FigureCanvas(ThreshAllCanvas.figure)
        ThreshAllCanvas.toolbar = NavigationToolbar(ThreshAllCanvas.canvas, self)
        ThreshAllCanvas.axes = ThreshAllCanvas.figure.add_subplot(1,1,1)

        ThreshAllCanvas.in_occ = []
        ThreshAllCanvas.in_PrDs = []
        ThreshAllCanvas.out_occ = []
        ThreshAllCanvas.out_PrDs = []
        for i in range(0,len(P1.all_PrDs)):
            if P1.all_occ[i] >= p.PDsizeThL:
                ThreshAllCanvas.in_occ.append(P1.all_occ[i])
                ThreshAllCanvas.in_PrDs.append(P1.all_PrDs[i])
            else:
                ThreshAllCanvas.out_occ.append(P1.all_occ[i])
                ThreshAllCanvas.out_PrDs.append(P1.all_PrDs[i])

        ThreshAllCanvas.axes.bar(ThreshAllCanvas.in_PrDs, ThreshAllCanvas.in_occ, align='center',\
                                 edgecolor='none', color='#1f77b4', snap=False)
        ThreshAllCanvas.axes.bar(ThreshAllCanvas.out_PrDs, ThreshAllCanvas.out_occ, align='center',\
                                 edgecolor='none', color='#1f77b4', snap=False)

        ThreshAllCanvas.xlimLo = ThreshAllCanvas.axes.get_xlim()[0]
        ThreshAllCanvas.xlimHi = ThreshAllCanvas.axes.get_xlim()[1]

        ThreshAllCanvas.lineL, = self.axes.plot([],[], color='#d62728', linestyle='-', linewidth=.5, label='Low Threshold') #red
        ThreshAllCanvas.lineH, = self.axes.plot([],[], color='#2ca02c', linestyle='-', linewidth=.5, label='High Threshold') #green
        x = np.arange(ThreshAllCanvas.xlimLo, ThreshAllCanvas.xlimHi+1)
        ThreshAllCanvas.lineL.set_data(x, p.PDsizeThL)
        ThreshAllCanvas.lineH.set_data(x, p.PDsizeThH)

        ThreshAllCanvas.axes.axvline(len(P1.all_PrDs)+1, color='#7f7f7f', linestyle='-', linewidth=.5)

        #ThreshAllCanvas.axes.legend(prop={'size': 6})#, loc='best')
        for tick in ThreshAllCanvas.axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in ThreshAllCanvas.axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        ThreshAllCanvas.axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        ThreshAllCanvas.axes.set_xlim(xmin=1, xmax=len(P1.all_PrDs))
        ThreshAllCanvas.axes.set_ylim(ymin=0, ymax=ThreshAllCanvas.thresh_high+20)
        ThreshAllCanvas.axes.set_xlabel('PD Numbers', fontsize=6)
        ThreshAllCanvas.axes.set_ylabel('Occupancy', fontsize=6)
        #ThreshAllCanvas.axes.autoscale()

        ThreshAllCanvas.canvas.draw()

        # threshold inputs:
        def choose_thresholds():
            ThreshAllCanvas.thresh_low = int(ThreshAllCanvas.entry_low.value())
            ThreshAllCanvas.thresh_high = int(ThreshAllCanvas.entry_high.value())
            self.replot()

        label_low = QtGui.QLabel('Low Threshold:')
        label_low.setFont(font_standard)

        label_high = QtGui.QLabel('High Threshold:')
        label_high.setFont(font_standard)

        ThreshAllCanvas.entry_prd = QtGui.QDoubleSpinBox(self)
        ThreshAllCanvas.entry_prd.setButtonSymbols(QtGui.QAbstractSpinBox.NoButtons)
        ThreshAllCanvas.entry_prd.setDecimals(0)
        ThreshAllCanvas.entry_prd.setMaximum(10000)
        ThreshAllCanvas.entry_prd.setDisabled(True)
        ThreshAllCanvas.entry_prd.setValue(len(ThreshAllCanvas.in_PrDs))

        ThreshAllCanvas.btn_update = QtGui.QPushButton('Update Thresholds')
        ThreshAllCanvas.btn_update.clicked.connect(self.confirmThresh)

        ThreshAllCanvas.btn_update.setDefault(False)
        ThreshAllCanvas.btn_update.setAutoDefault(False)

        ThreshAllCanvas.entry_low = QtGui.QDoubleSpinBox(self)
        ThreshAllCanvas.entry_low.setDecimals(0)
        ThreshAllCanvas.entry_low.setMinimum(90)
        ThreshAllCanvas.entry_low.setMaximum(np.amax(P1.all_occ))
        ThreshAllCanvas.entry_low.setValue(int(p.PDsizeThL))

        ThreshAllCanvas.entry_high = QtGui.QDoubleSpinBox(self)
        ThreshAllCanvas.entry_high.setDecimals(0)
        ThreshAllCanvas.entry_high.setMinimum(90)
        ThreshAllCanvas.entry_high.setMaximum(10000)
        ThreshAllCanvas.entry_high.setValue(int(p.PDsizeThH))

        ThreshAllCanvas.entry_low.valueChanged.connect(choose_thresholds)
        ThreshAllCanvas.entry_high.valueChanged.connect(choose_thresholds)

        if (ThreshAllCanvas.thresh_low == p.PDsizeThL and ThreshAllCanvas.thresh_high == p.PDsizeThH) or\
            ThreshAllCanvas.thresh_low == p.PDsizeThL and ThreshAllCanvas.thresh_high == np.amax(P1.all_occ)+1:
                ThreshAllCanvas.btn_update.setDisabled(True)
                ThreshAllCanvas.progBar.setVisible(False)
                ThreshAllCanvas.confirmed = 1
        else:
            ThreshAllCanvas.btn_update.setDisabled(False)
            ThreshAllCanvas.progBar.setVisible(True)
            ThreshAllCanvas.progBar.setValue(0)
            ThreshAllCanvas.confirmed = 0

        label_prd = QtGui.QLabel('Number of PDs:')
        label_prd.setFont(font_standard)

        layout = QtGui.QGridLayout()
        layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        layout.addWidget(ThreshAllCanvas.toolbar, 0, 0, 1, 8, QtCore.Qt.AlignVCenter)
        layout.addWidget(ThreshAllCanvas.canvas, 1, 0, 1, 8, QtCore.Qt.AlignVCenter)
        layout.addWidget(ThreshAllCanvas.progBar, 2, 0, 1, 8, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_low, 3, 0, 1, 1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout.addWidget(ThreshAllCanvas.entry_low, 3, 1, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_high, 3, 2, 1, 1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout.addWidget(ThreshAllCanvas.entry_high, 3, 3, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(label_prd, 3, 4, 1, 1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout.addWidget(ThreshAllCanvas.entry_prd, 3, 5, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(ThreshAllCanvas.btn_update, 3, 6, 1, 1, QtCore.Qt.AlignVCenter)
        self.setLayout(layout)

        if p.resProj > 0 or P2.noReturnFinal == True:
            ThreshAllCanvas.btn_update.setDisabled(True)
            ThreshAllCanvas.entry_low.setDisabled(True)
            ThreshAllCanvas.entry_high.setDisabled(True)


    def replot(self):
        x = np.arange(ThreshAllCanvas.xlimLo, ThreshAllCanvas.xlimHi+1)
        ThreshAllCanvas.lineL.set_data(x, ThreshAllCanvas.thresh_low)
        ThreshAllCanvas.lineH.set_data(x, ThreshAllCanvas.thresh_high)
        ThreshAllCanvas.axes.set_ylim(ymin=0, ymax=ThreshAllCanvas.thresh_high+20)

        ThreshAllCanvas.canvas.draw()

        if (ThreshAllCanvas.thresh_low == p.PDsizeThL and ThreshAllCanvas.thresh_high == p.PDsizeThH) or\
            ThreshAllCanvas.thresh_low == p.PDsizeThL and ThreshAllCanvas.thresh_high == np.amax(P1.all_occ)+1:
                ThreshAllCanvas.btn_update.setDisabled(True)
                ThreshAllCanvas.progBar.setVisible(False)
                ThreshAllCanvas.confirmed = 1

        else:
            ThreshAllCanvas.btn_update.setDisabled(False)
            ThreshAllCanvas.progBar.setValue(0)
            ThreshAllCanvas.progBar.setVisible(True)
            ThreshAllCanvas.confirmed = 0

        ThreshAllCanvas.in_occ = []
        ThreshAllCanvas.in_PrDs = []
        ThreshAllCanvas.out_occ = []
        ThreshAllCanvas.out_PrDs = []

        for i in range(0, len(P1.all_PrDs)):
            if P1.all_occ[i] >= ThreshAllCanvas.thresh_low:
                ThreshAllCanvas.in_occ.append(P1.all_occ[i])
                ThreshAllCanvas.in_PrDs.append(i)
            else:
                ThreshAllCanvas.out_occ.append(P1.all_occ[i])
                ThreshAllCanvas.out_PrDs.append(i)

        ThreshAllCanvas.entry_prd.setValue(len(ThreshAllCanvas.in_PrDs))

        if p.resProj > 0:
            ThreshAllCanvas.btn_update.setDisabled(True)
            ThreshAllCanvas.entry_low.setDisabled(True)
            ThreshAllCanvas.entry_high.setDisabled(True)

    def confirmThresh(self):
        if len(ThreshAllCanvas.in_PrDs) > 2:        
            p.PDsizeThL = ThreshAllCanvas.thresh_low
            p.PDsizeThH = ThreshAllCanvas.thresh_high
            self.DataStart()
        else:
            msg = 'A minimum of 3 PDs are required.\
                    <br /><br />\
                    Please select new thresholds before updating.'
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Error' % progname)
            box.setText('<b>Input Error</b>')
            box.setFont(font_standard)
            box.setIcon(QtGui.QMessageBox.Warning)
            box.setInformativeText(msg)
            box.setStandardButtons(QtGui.QMessageBox.Ok)        
            ret = box.exec_()

    def DataStart(self):
        ThreshAllCanvas.btn_update.setDisabled(True)
        ThreshAllCanvas.confirmed = 1
        #ThreshAllCanvas.progBar.setRange(0,0)
        ThreshAllCanvas.progBar.setValue(50)
        self.DataTask.start()

    def DataFinished(self):
        time.sleep(1) #wait time to ensure files are completely written
        fname = open(os.path.join(P1.user_directory,'outputs_{}/selecGCs'.format(p.proj_name)), 'rb')
        data = pickle.load(fname)
        P1.S2 = data['S2']
        P1.CG = data['CG']
        fname.close()
        #P2.viz1.update_scene1()

        # re-threshold bins:
        P1.thresh_PrDs = []
        P1.thresh_occ = []
        pd = 1
        for i in range(0, len(P1.all_PrDs)):
            if P1.all_occ[i] >= p.PDsizeThL:
                P1.thresh_PrDs.append(pd)
                if P1.all_occ[i] > p.PDsizeThH:
                    P1.thresh_occ.append(p.PDsizeThH)
                else:
                    P1.thresh_occ.append(P1.all_occ[i])
                pd += 1

        ThreshAllCanvas.progBar.setValue(100)

        print('')
        print('New thresholds set:')
        print('high:',p.PDsizeThH)
        print('low:',p.PDsizeThL)
        print('')

        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Set Thresholds' % progname)
        box.setIcon(QtGui.QMessageBox.Information)
        box.setText('<b>Thresholding Complete</b>')
        box.setFont(font_standard)
        msg = 'New high and low thresholds have been set.'
        box.setStandardButtons(QtGui.QMessageBox.Ok)
        box.setInformativeText(msg)
        reply = box.exec_()


class DataThread(QtCore.QThread):
    DataFinished = QtCore.pyqtSignal()
    def run(self):
        set_params.op(0) #send new GUI data to parameters file
        Data.op(P1.user_alignment)
        self.DataFinished.emit()


class ThreshFinalCanvas(QtGui.QDialog):
    def __init__(self, parent=None):
        super(ThreshFinalCanvas, self).__init__(parent)
        self.left = 10
        self.top = 10

        # create canvas and plot data:
        ThreshFinalCanvas.figure = Figure(dpi=200)
        ThreshFinalCanvas.canvas = FigureCanvas(ThreshFinalCanvas.figure)
        #ThreshFinalCanvas.toolbar = NavigationToolbar(ThreshFinalCanvas.canvas, self)
        ThreshFinalCanvas.axes = ThreshFinalCanvas.figure.add_subplot(1,1,1, polar=True)
        thresh = ThreshFinalCanvas.axes.scatter([0], [0], edgecolor='k', linewidth=.5, c=[0], s=5, cmap=cm.hsv) #empty for init
        ThreshFinalCanvas.cbar = ThreshFinalCanvas.figure.colorbar(thresh, pad=0.1)
        #ThreshFinalCanvas.axes.autoscale()

        # thetas = [0,45,90,135,180,225,270,315] #in same order as labels below (ref only)
        theta_labels = ['%s180%s'%(u"\u00B1",u"\u00b0"),'-135%s'%(u"\u00b0"),'-90%s'%(u"\u00b0"),'-45%s'%(u"\u00b0"),
                        '0%s'%(u"\u00b0"),'45%s'%(u"\u00b0"),'90%s'%(u"\u00b0"),'135%s'%(u"\u00b0")]
        ThreshFinalCanvas.axes.set_ylim(0,180)
        ThreshFinalCanvas.axes.set_yticks(np.arange(0,180,20))
        ThreshFinalCanvas.axes.set_xticklabels(theta_labels)
        for tick in ThreshFinalCanvas.axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in ThreshFinalCanvas.axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        ThreshFinalCanvas.axes.grid(alpha=0.2)

        ThreshFinalCanvas.layout = QtGui.QGridLayout()
        #layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        #ThreshFinalCanvas.layout.addWidget(ThreshFinalCanvas.toolbar, 0, 0, 1, 8, QtCore.Qt.AlignVCenter)
        ThreshFinalCanvas.layout.addWidget(ThreshFinalCanvas.canvas, 1, 0, 1, 8, QtCore.Qt.AlignVCenter)
        self.setLayout(ThreshFinalCanvas.layout)


class OccHistCanvas(QtGui.QDialog):
    numBins = 100

    def __init__(self, parent=None):
        super(OccHistCanvas, self).__init__(parent)
        self.left = 10
        self.top = 10

        # create canvas and plot data:
        OccHistCanvas.figure = Figure(dpi=200)
        OccHistCanvas.canvas = FigureCanvas(OccHistCanvas.figure)
        OccHistCanvas.toolbar = NavigationToolbar(OccHistCanvas.canvas, self)
        OccHistCanvas.axes = OccHistCanvas.figure.add_subplot(1,1,1)
        OccHistCanvas.axes.ticklabel_format(useOffset=False, style='plain')
        OccHistCanvas.axes.get_xaxis().get_major_formatter().set_scientific(False)

        layout = QtGui.QGridLayout()

        OccHistCanvas.label_bins = QtGui.QLabel('Histogram bins:')
        OccHistCanvas.label_bins.setFont(font_standard)
        OccHistCanvas.label_bins.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        OccHistCanvas.entry_bins = QtGui.QDoubleSpinBox(self)
        OccHistCanvas.entry_bins.setDecimals(0)
        OccHistCanvas.entry_bins.setMinimum(2)
        OccHistCanvas.entry_bins.setValue(int(len(set(P1.thresh_occ))/2.))
        OccHistCanvas.entry_bins.setMaximum(len(set(P1.thresh_occ)))
        OccHistCanvas.entry_bins.setSuffix(' / %s' % len(set(P1.thresh_occ))) #number of unique occupancies
        OccHistCanvas.entry_bins.valueChanged.connect(self.change_bins)
        OccHistCanvas.entry_bins.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        layout.addWidget(OccHistCanvas.toolbar, 0, 0, 1, 8, QtCore.Qt.AlignVCenter)
        layout.addWidget(OccHistCanvas.canvas, 1, 0, 1, 8, QtCore.Qt.AlignVCenter)
        layout.addWidget(OccHistCanvas.label_bins, 2, 1, 1, 1)
        layout.addWidget(OccHistCanvas.entry_bins, 2, 2, 1, 1)
        self.setLayout(layout)


    def change_bins(self):
        OccHistCanvas.numBins = int(OccHistCanvas.entry_bins.value())
        # re-threshold bins:
        temp_PrDs = []
        temp_occ = []
        for i in range(0, len(P1.all_PrDs)):
            if P1.all_occ[i] >= ThreshAllCanvas.thresh_low:
                temp_PrDs.append(i+1)
                if P1.all_occ[i] > p.PDsizeThH:
                    temp_occ.append(p.PDsizeThH)
                else:
                    temp_occ.append(P1.all_occ[i])

        # replot OccHistCanvas:
        OccHistCanvas.axes.clear()
        counts, bins, bars = OccHistCanvas.axes.hist(temp_occ, bins=int(OccHistCanvas.numBins), align='left',\
                                                     edgecolor='w', linewidth=1, color='#1f77b4') #C0

        OccHistCanvas.axes.set_xticks(bins)#[:-1]
        OccHistCanvas.axes.set_title('PD Occupancy Distribution', fontsize=6)
        OccHistCanvas.axes.set_xlabel('PD Occupancy', fontsize=5)
        OccHistCanvas.axes.set_ylabel('Number of PDs', fontsize=5)
        OccHistCanvas.axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        OccHistCanvas.axes.get_xaxis().get_major_formatter().set_scientific(False)
        OccHistCanvas.axes.ticklabel_format(useOffset=False, style='plain')

        for tick in OccHistCanvas.axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in OccHistCanvas.axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(4)

        OccHistCanvas.axes.autoscale()
        OccHistCanvas.figure.canvas.draw()
        
        
################################################################################
# overhead GUI control: 

class MainWindow(QtGui.QMainWindow):
    inputsFile = ''
    restoreComplete = 1 #sets to 1 once previous project fully restored
    S2_rho = 50
    S2_scale = 1.
    S2_iso = 3
    # keep track when above variables are altered:
    S2_scale_prev = 1.
    S2_iso_prev = 3

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setWindowTitle(progname)

        tab1 = P1(self)
        tab2 = P2(self)
        tab3 = P3(self)
        tab4 = P4(self)
        tab5 = P5(self)
        tab6 = P6(self)
        #tab7 = P7(self)

        global tabs
        tabs = QtGui.QTabWidget(self)
        tabs.resize(250,150)

        tabs.addTab(tab1, 'Import')
        tabs.addTab(tab2, 'Distribution')
        tabs.addTab(tab3, 'Embedding')
        tabs.addTab(tab4, 'Eigenvectors')
        tabs.addTab(tab5, 'Compilation')
        tabs.addTab(tab6, 'Energetics')
        #tabs.addTab(tab7, 'Dynamics')

        # =====================================================================
        # GUI LAYOUT OPTIONS START
        # =====================================================================
        # choose A or B only:
        if 0: # A: if 1 = tabbed windows only
            self.setCentralWidget(tabs)
            
        if 1: # B: if 1 = scrollable windows + tabs
            self.groupscroll = QtGui.QHBoxLayout()
            self.groupscrollbox = QtGui.QGroupBox()

            self.MVB = QtGui.QVBoxLayout()
            self.MVB.addWidget(tabs)

            scroll = QtGui.QScrollArea()
            widget = QtGui.QWidget(self)
            widget.setLayout(QtGui.QHBoxLayout())
            widget.layout().addWidget(self.groupscrollbox)
            scroll.setWidget(widget)
            scroll.setWidgetResizable(True)
            self.groupscrollbox.setLayout(self.MVB)
            self.groupscroll.addWidget(scroll)
            self.setCentralWidget(scroll)

            #self.showMaximized() #start in fullscreen mode
            sizeObject = QtGui.QDesktopWidget().screenGeometry(-1) #user screen size
            self.setMinimumSize(500,300)
            self.setMaximumSize(sizeObject.width(),sizeObject.height())
            self.resize(sizeObject.width()-100,sizeObject.height()-100) #slightly less than full-screen
        # =====================================================================
        # GUI LAYOUT OPTIONS END
        # =====================================================================

        tab1.button_toP2.clicked.connect(self.gotoP2)
        tab2.button_binPart.clicked.connect(self.gotoP3)        
        self.show()

        # File Menu:
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction('&About', self.about)
        fileMenu.addSeparator()
        fileMenu.addAction('&Restart', self.fileRestart)
        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction('&Import', self.guide_import)
        helpMenu.addAction('&Distribution', self.guide_orientations)
        helpMenu.addAction('&Embedding', self.guide_embedding)
        helpMenu.addAction('&Eigenvectors', self.guide_eigenvectors)
        helpMenu.addAction('&Compilation', self.guide_outputs)
        dynMenu = helpMenu.addMenu('&Energetics')
        d1Menu = dynMenu.addMenu('&1D Energy Path')
        d1Menu.addAction('&Energy Path Analysis', self.guide_dyn_1d)
        d2Menu = dynMenu.addMenu('&2D Energy Landscape')
        d2Menu.addAction('&Custom Path Analysis', self.guide_dyn_custom)
        d2Menu.addAction('&Least Action Path Analysis', self.guide_dyn_action)
        helpMenu.addSeparator()
        helpMenu.addAction('&Camera Controls', self.cameraInfo)

        # disables nonlinear user-access to tabs:
        tabs.setTabEnabled(1, False)
        tabs.setTabEnabled(2, False)
        tabs.setTabEnabled(3, False)
        tabs.setTabEnabled(4, False)
        tabs.setTabEnabled(5, False)
        tabs.setTabEnabled(6, False)
        tabs.setTabEnabled(7, False)

        global noReturn
        def noReturn():
            # DISABLES PAGE 1:
            tab1.button_toP2.setDisabled(True)
            P2.noReturnFinal = True
            tab1.entry_avgVol.setDisabled(True)
            tab1.entry_imgStack.setDisabled(True)
            tab1.entry_align.setDisabled(True)
            tab1.entry_name.setDisabled(True)
            tab1.entry_pixel.setDisabled(True)
            tab1.entry_objDiam.setDisabled(True)
            tab1.entry_resolution.setDisabled(True)
            tab1.entry_aperture.setDisabled(True)
            tab1.button_browse1.setDisabled(True)
            tab1.button_browse2.setDisabled(True)
            tab1.button_browse3.setDisabled(True)
            tab1.button_browse4.setDisabled(True)
            tab1.button_browseM.setDisabled(True)
            tab2.button_binPart.setDisabled(True)
            tab2.button_binPart.setText('Particle Binning Complete')

        tabs.currentChanged.connect(self.onTabChange) #signal for tabs changed via direct click

        # =====================================================================
        # Branch to continue progress from previous session:
        # =====================================================================

        MainWindow.inputsFile = ''
        if userInput != 'empty':
            MainWindow.inputsFile = os.path.join(pyDir, 'params_' + userInput + '.pkl')
        
        elif userInput == 'empty':
            msg = 'Would you like to initiate a new project or return to progress already made on a previous one?'
            box = QtGui.QMessageBox()
            box.setWindowTitle('%s Startup' % progname)
            box.setText('<b>Project Initiation</b>')
            iconDir = os.path.join(pyDir, 'icons/70x70.png')
            if iconDir:
                box.setIconPixmap(QtGui.QPixmap(iconDir))
            box.setFont(font_standard)
            newBtn = box.addButton('   New Session   ', QtGui.QMessageBox.NoRole)
            restoreBtn = box.addButton(' Restore Session ', QtGui.QMessageBox.YesRole)
            box.setDefaultButton(newBtn)
            box.setInformativeText(msg)
            box.exec_()
            if box.clickedButton() == restoreBtn: #return to previous project
                box = QtGui.QMessageBox(self)
                box.setWindowTitle('%s Startup' % progname)
                box.setText('<b>Restore Previous Session</b>')
                box.setFont(font_standard)
                box.setIcon(QtGui.QMessageBox.Information)
                box.setInformativeText('On the following screen, select the\
                                        <i>params.pkl</i> file for the previously\
                                        initiated project.\
                                        <br /><br />\
                                        This file is uniquely created within the main directory\
                                        of each project.')
                box.setStandardButtons(QtGui.QMessageBox.Ok)
                box.setDefaultButton(QtGui.QMessageBox.Ok)
                ret = box.exec_()

                MainWindow.inputsFile = QtGui.QFileDialog.getOpenFileName(self, 'Choose previous <i>params.pkl</i> file', '',
                                                         ('Data Files (*.pkl)'))[0]

        if MainWindow.inputsFile:

            if MainWindow.inputsFile.endswith('.pkl'):
                print('Loading user data...')
                try:
                    fname_front = MainWindow.inputsFile.split('params_',1)[1]
                    fname_sans = os.path.splitext(fname_front)[0]
                    p.proj_name = fname_sans #update p.py with correct project name (not the GUI init)
                    set_params.op(1) #read in params from user parameters file

                    # =========================================================
                    # DEVELOPER OPTION TO SHORTCUT RESUME PROGRESS:
                    # =========================================================
                    # p.resProj = {int}
                    # =========================================================
                    # DEVELOPER OPTION TO OVERWRITE PREVIOUS DIRECTORY LOCATION:
                    # =========================================================
                    # p.user_dir = '/mnt/Data2/evanseitz/manifold_1D-v2' #'path/to/file'
                    # =========================================================

                    print('\nUpdating import tab...')
                    print('\tproject name: %s' % (p.proj_name))
                    print('\tuser directory: %s' % (p.user_dir))
                    print('\tresume progress: %s' % (p.resProj))
                    print('\trelion data: %s' % (p.relion_data))

                    P1.user_name = p.proj_name
                    P1.entry_name.setText(p.proj_name)
                    
                    if not os.path.exists(p.avg_vol_file):
                        msg = 'The previous <b>Average Volume</b> file can not be found.\
                                <br /><br />\
                                Please update ManifoldEM with its current location.'
                        box = QtGui.QMessageBox(self)
                        box.setWindowTitle('%s Error' % progname)
                        box.setText('<b>Input Error</b>')
                        box.setFont(font_standard)
                        box.setIcon(QtGui.QMessageBox.Warning)
                        box.setInformativeText(msg)
                        box.setStandardButtons(QtGui.QMessageBox.Ok)        
                        ret = box.exec_()

                        fileName = QtGui.QFileDialog.getOpenFileName(self, 'Choose Data File', '',
                                                                     ('Data Files (*.mrc)'))[0]
                        if fileName:
                            if fileName.endswith('.mrc'):
                                p.avg_vol_file = fileName
                                
                    P1.user_volume = p.avg_vol_file    
                    P1.entry_avgVol.setText(P1.user_volume)
                    with mrcfile.open(P1.user_volume, mode='r') as mrc:
                        P1.df_vol = mrc.data
                    print('\tavg volume: %s' % (p.avg_vol_file))

                    if not os.path.exists(p.img_stack_file):
                        msg = 'The previous <b>Image Stack</b> file can not be found.\
                                <br /><br />\
                                Please update ManifoldEM with its current location.'
                        box = QtGui.QMessageBox(self)
                        box.setWindowTitle('%s Error' % progname)
                        box.setText('<b>Input Error</b>')
                        box.setFont(font_standard)
                        box.setIcon(QtGui.QMessageBox.Warning)
                        box.setInformativeText(msg)
                        box.setStandardButtons(QtGui.QMessageBox.Ok)        
                        ret = box.exec_()

                        fileName = QtGui.QFileDialog.getOpenFileName(self, 'Choose Data File', '',
                                                                     ('Data Files (*.mrcs)'))[0]
                        if fileName:
                            if fileName.endswith('.mrcs'):
                                p.img_stack_file = fileName

                    if p.img_stack_file.endswith('.mrcs'):
                        mrc = mrcfile.mmap(p.img_stack_file, mode='r+')
                        mrc.set_image_stack()
                    P1.user_stack = p.img_stack_file
                    P1.entry_imgStack.setText(p.img_stack_file)
                    print('\timage stack: %s' % (p.img_stack_file))
                    
                    if not os.path.exists(p.img_stack_file):
                        msg = 'The previous <b>Alignment/b> file can not be found.\
                                <br /><br />\
                                Please update ManifoldEM with its current location.'
                        box = QtGui.QMessageBox(self)
                        box.setWindowTitle('%s Error' % progname)
                        box.setText('<b>Input Error</b>')
                        box.setFont(font_standard)
                        box.setIcon(QtGui.QMessageBox.Warning)
                        box.setInformativeText(msg)
                        box.setStandardButtons(QtGui.QMessageBox.Ok)        
                        ret = box.exec_()
    
                        fileName = QtGui.QFileDialog.getOpenFileName(self, 'Choose Data File', '',
                                                                     ('Data Files (*.star)'))[0]
                        if fileName:
                            if fileName.endswith('.star'):
                                p.align_param_file = fileName
                    
                    P1.user_alignment = p.align_param_file
                    P1.entry_align.setText(P1.user_alignment)
                    print('\talignment: %s' % (p.align_param_file))
                    
                    if not os.path.exists(p.mask_vol_file) and p.mask_vol_file != '':
                        msg = 'The previous <b>Mask Volume</b> file can not be found.\
                                <br /><br />\
                                Please update ManifoldEM with its current location.'
                        box = QtGui.QMessageBox(self)
                        box.setWindowTitle('%s Error' % progname)
                        box.setText('<b>Input Error</b>')
                        box.setFont(font_standard)
                        box.setIcon(QtGui.QMessageBox.Warning)
                        box.setInformativeText(msg)
                        box.setStandardButtons(QtGui.QMessageBox.Ok)        
                        ret = box.exec_()
    
                        fileName = QtGui.QFileDialog.getOpenFileName(self, 'Choose Data File', '',
                                                                     ('Data Files (*.mrc)'))[0]
                        if fileName:
                            if fileName.endswith('.mrc'):
                                p.mask_vol_file = fileName
                    
                    P1.user_mask = p.mask_vol_file
                    P1.entry_maskVol.setText(P1.user_mask)
                    print('\tmask volume: %s' % (p.mask_vol_file))
                    
                    P1.entry_pixel.setValue(p.pix_size)
                    print('\tpixel size: %s' % (p.pix_size))

                    P1.entry_objDiam.setValue(p.obj_diam)
                    print('\tdiameter: %s' % (p.obj_diam))

                    P1.entry_resolution.setValue(p.resol_est)
                    print('\tresolution: %s' % (p.resol_est))

                    P1.entry_aperture.setValue(p.ap_index)
                    print('\taperture index: %s' % (p.ap_index))
                    
                    P1.user_shannon = p.sh
                    P1.entry_shannon.setValue(P1.user_shannon)
                    print('\tshannon angle: %s' % (p.sh))
                    
                    P1.user_width = p.ang_width
                    P1.entry_angWidth.setValue(P1.user_width)
                    print('\tangle width: %s' % (p.ang_width))

                    P1.S2rescale = p.S2rescale
                    MainWindow.S2_scale = p.S2rescale
                    MainWindow.S2_scale_prev = p.S2rescale
                    print('\tS2 scale: %s' % (p.S2rescale))
                    
                    MainWindow.S2_iso = p.S2iso
                    MainWindow.S2_iso_prev = p.S2iso
                    print('\tS2 isosurface level: %s' % (p.S2iso))

                    print('\nUpdating distribution tab...')

                    p.create_dir()
                    p.tess_file = os.path.join(P1.user_directory,'outputs_{}/selecGCs'.format(p.proj_name))
                    p.nowTime_file = os.path.join(P1.user_directory,'outputs_{}/nowTime'.format(p.proj_name))

                    fname = open(os.path.join(P1.user_directory,'outputs_{}/selecGCs'.format(p.proj_name)), 'rb')
                    data = pickle.load(fname)
                    P1.S2 = data['S2']
                    P1.CG = data['CG']                    

                    print('\twindow size: %s' % (p.nPix))
                    print('\tspherical aberration: %s' % (p.Cs))
                    print('\tvoltage: %s' % (p.EkV))
                    print('\tamplitude contrast: %s' % (p.AmpContrast))

                    # =========================================================
                    # Calculate PrD occupancies for figure on P2:
                    # =========================================================
                    ThreshAllCanvas.thresh_low = int(p.PDsizeThL)
                    ThreshAllCanvas.thresh_high = int(p.PDsizeThH)
                    print('\tlow threshold: %s' % p.PDsizeThL)
                    print('\thigh threshold: %s' % p.PDsizeThH)
                    print('\tthresholding PDs...')
                    fname = open(os.path.join(P1.user_directory,'outputs_{}/selecGCs'.format(p.proj_name)), 'rb')
                    data = pickle.load(fname)
                    # all tessellated bins:
                    totalPrDs = int(np.shape(data['CG1'])[0])
                    mid = np.floor(np.shape(data['CG1'])[0]/2)
                    NC1 = data['NC'][:int(mid)]
                    NC2 = data['NC'][int(mid):]

                    P1.all_PrDs = []
                    P1.all_occ = []
                    P1.thresh_PrDs = []
                    P1.thresh_occ = []
                    if len(NC1) >= len(NC2): #first half of S2
                        pd_all = 1
                        pd = 1
                        for i in NC1:
                            P1.all_PrDs.append(pd_all)
                            P1.all_occ.append(i)
                            if i >= p.PDsizeThL:
                                P1.thresh_PrDs.append(pd)
                                if i > p.PDsizeThH:
                                    P1.thresh_occ.append(p.PDsizeThH)
                                else:
                                    P1.thresh_occ.append(i)
                                pd += 1
                            pd_all += 1
                    else: #second half of S2
                        pd_all = 1
                        pd = 1
                        for i in NC2:
                            P1.all_PrDs.append(pd_all)
                            P1.all_occ.append(i)
                            if i >= p.PDsizeThL:
                                P1.thresh_PrDs.append(pd)
                                if i > p.PDsizeThH:
                                    P1.thresh_occ.append(p.PDsizeThH)
                                else:
                                    P1.thresh_occ.append(i)
                                pd += 1
                            pd_all += 1

                    # read points from tesselated sphere:
                    PrD_map1 = os.path.join(P1.user_directory,'outputs_{}/topos/Euler_PrD/PrD_map1.txt'.format(p.proj_name))
                    PrD_map1_eul = []
                    with open(PrD_map1) as values:
                        for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                            PrD_map1_eul.append(column)

                    P1.all_phi = PrD_map1_eul[2]
                    P1.all_theta = PrD_map1_eul[1]

                    # ad hoc ratios to make sure S2 volume doesn't freeze-out due to too many particles to plot:
                    ratio1 = float(sum(P1.all_occ))/2000
                    ratio2 = float(sum(P1.all_occ))/5000
                    MainWindow.S2_rho = min(P1.S2_density_all, key=lambda x:abs(x-ratio1))
                    P1.S2_density_all = list(filter(lambda a: a < int(sum(P1.all_occ)), P1.S2_density_all))
                    P1.S2_density_all = list(filter(lambda a: a > int(ratio2), P1.S2_density_all))
                    
                    P2.viz1.update_S2_density_all()
                    P2.viz1.update_S2_params()
                    print('\ttotal particles: %s' % sum(P1.all_occ))
                    # update distribution tab:
                    print('\trendering visualizations...')

                    fname.close()
                    
                    # =========================================================
                    # Calculate total PDs:
                    # =========================================================
                    PrD_map0 = os.path.join(P1.user_directory,'outputs_{}/topos/Euler_PrD/PrD_map.txt'.format(p.proj_name))
                    PrD_map0_eul = []
                    with open(PrD_map0) as values:
                        for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                            PrD_map0_eul.append(column)

                    P3.PrD_total = len(PrD_map0_eul[1])
                    print('\ttotal PDs: %s' % P3.PrD_total)
                    fname.close()
                    # =========================================================

                    P3.dictGen = False
                    P4.user_PrD = 1
                    P4.PrD_hist = 1

                    MainWindow.restoreComplete = 0 #previous project restoration in progress
                    
                    P2.viz1.update_scene2()
                    P2.viz1.update_scene1()

                    if p.resProj == 1: #user has confirmed Data.py entries, but not (or only partially) started GetDistances
                        tabs.setTabEnabled(1, True)
                        tabs.setTabEnabled(2, True)
                        noReturn()
                        P3.user_processors = p.ncpu
                        P3.entry_proc.setValue(p.ncpu)
                        P3.user_psi = p.num_psis
                        P3.entry_psi.setValue(p.num_psis)
                        tabs.setCurrentIndex(2)

                    if p.resProj == 2: #processors & eigenvectors entered; GetDistancesS2 complete
                        print('\nUpdating embedding tab...')
                        P3.user_processors = p.ncpu
                        P3.entry_proc.setValue(p.ncpu)
                        P3.user_psi = p.num_psis
                        P3.entry_psi.setValue(p.num_psis)
                        P3.entry_psi.setDisabled(True)
                        P3.button_dist.setDisabled(True)
                        P3.button_dist.setText('Distance Calculation Complete')
                        P3.progress1.setValue(100)
                        P3.button_eig.setDisabled(False)
                        print('\tprocessors: %s' % (p.ncpu))
                        print('\teigenvectors: %s' % (p.num_psis))

                        tabs.setTabEnabled(1, True)
                        tabs.setTabEnabled(2, True)
                        noReturn()
                        tabs.setCurrentIndex(2)
                       
                    if p.resProj == 3: #18 ManifoldAnalysis complete
                        print('\nUpdating embedding tab...')
                        P3.user_processors = p.ncpu
                        P3.entry_proc.setValue(p.ncpu)
                        P3.user_psi = p.num_psis
                        P3.entry_psi.setValue(p.num_psis)
                        P3.entry_psi.setDisabled(True)
                        P3.button_dist.setDisabled(True)
                        P3.button_dist.setText('Distance Calculation Complete')
                        P3.progress1.setValue(100)
                        P3.button_eig.setDisabled(True)
                        P3.button_eig.setText('Embedding Complete')
                        P3.progress2.setValue(100)
                        P3.button_psi.setDisabled(False)
                        print('\tprocessors: %s' % (p.ncpu))
                        print('\teigenvectors: %s' % (p.num_psis))

                        tabs.setTabEnabled(1, True)
                        tabs.setTabEnabled(2, True)
                        noReturn()
                        tabs.setCurrentIndex(2)
                        
                    if p.resProj == 4: #19 PsiAnalysis complete
                        print('\nUpdating embedding tab...')
                        P3.user_processors = p.ncpu
                        P3.entry_proc.setValue(p.ncpu)
                        P3.user_psi = p.num_psis
                        P3.entry_psi.setValue(p.num_psis)
                        P3.entry_psi.setDisabled(True)
                        P3.button_dist.setDisabled(True)
                        P3.button_dist.setText('Distance Calculation Complete')
                        P3.progress1.setValue(100)
                        P3.button_eig.setDisabled(True)
                        P3.button_eig.setText('Embedding Complete')
                        P3.progress2.setValue(100)
                        P3.button_psi.setDisabled(True)
                        P3.button_psi.setText('Spectral Analysis Complete')
                        P3.progress3.setValue(100)
                        P3.button_nlsa.setDisabled(False)
                        print('\tprocessors: %s' % (p.ncpu))
                        print('\teigenvectors: %s' % (p.num_psis))

                        tabs.setTabEnabled(1, True)
                        tabs.setTabEnabled(2, True)
                        noReturn()
                        tabs.setCurrentIndex(2)

                    if p.resProj == 5: #20 NLSAmovie complete
                        print('\nUpdating embedding tab...')
                        P3.user_processors = p.ncpu
                        P3.entry_proc.setValue(p.ncpu)
                        P3.entry_proc.setDisabled(True)
                        P3.user_psi = (p.num_psis)
                        P3.entry_psi.setValue(p.num_psis)
                        P3.entry_psi.setDisabled(True)
                        P3.button_dist.setDisabled(True)
                        P3.button_dist.setText('Distance Calculation Complete')
                        P3.progress1.setValue(100)
                        P3.button_eig.setDisabled(True)
                        P3.button_eig.setText('Embedding Complete')
                        P3.progress2.setValue(100)
                        P3.button_psi.setDisabled(True)
                        P3.button_psi.setText('Spectral Analysis Complete')
                        P3.progress3.setValue(100)
                        P3.button_nlsa.setDisabled(True)
                        P3.button_nlsa.setText('2D Movies Complete')
                        P3.progress4.setValue(100)
                        P3.user_dimensions = p.dim
                        P3.entry_dim.setValue(p.dim)
                        print('\tprocessors: %s' % (p.ncpu))
                        print('\teigenvectors: %s' % (p.num_psis))
                        print('\tdimensions: %s' % (p.dim))

                        print('\nUpdating eigenvectors tab...')
                        tabs.setTabEnabled(1, True)
                        tabs.setTabEnabled(2, True)
                        tabs.setTabEnabled(3, True)
                        noReturn()
                        tabs.setCurrentIndex(2)
                        P3.button_toP4.setDisabled(False)
                        
                        # keep track of number of times user has re-embedded per PD:
                        P4.origEmbedFile = os.path.join(P1.user_directory, 'outputs_{}/topos/Euler_PrD/PrD_embeds.txt'.format(p.proj_name))
                        if os.path.isfile(P4.origEmbedFile) is False:
                            for i in range(P3.PrD_total):
                                P4.origEmbed.append(int(1)) #1 is True = PD has its original embedding
                            np.savetxt(P4.origEmbedFile, P4.origEmbed, fmt='%i')          
                        
                        else:
                            data = []
                            with open(P4.origEmbedFile) as values:
                                for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                                    data.append(column)
                            P4.origEmbed = []
                            idx = 0
                            for i in data[0]:
                                P4.origEmbed.append(int(i))
                                idx += 1
                        
                        P3.button_toP4.click()
                        # fix to make sure first PD's reaction coordinates not the same (and thus clickable) on startup:
                        if P4.reactCoord1All[1].value() == P4.reactCoord2All[1].value():
                            if P4.reactCoord1All[1].value() > 1:
                                P4.reactCoord2All[1].setValue(1)
                            elif P4.reactCoord1All[1].value() == 1:
                                P4.reactCoord2All[1].setValue(2)

                    if p.resProj >= 6: #Dimensions, Conformational Coordinates, etc. ('Compile Results' clicked)
                        print('\nUpdating embedding tab...')

                        P3.user_processors = p.ncpu
                        P3.entry_proc.setValue(p.ncpu)
                        P3.entry_proc.setDisabled(True)
                        
                        P3.user_psi = p.num_psis
                        P3.entry_psi.setValue(p.num_psis)
                        P3.entry_psi.setDisabled(True)
                        
                        P3.user_dimensions = p.dim
                        P3.entry_dim.setValue(p.dim)
                        print('\tprocessors: %s' % (p.ncpu))
                        print('\teigenvectors: %s' % (p.num_psis))
                        print('\tdimensions: %s' % (p.dim))

                        Erg1dMain.chooseCC.setCurrentIndex(0)
                        if P3.user_dimensions == 1: 
                            Erg1dMain.chooseCC.setDisabled(True) #ZULU check this
                        else:
                            Erg1dMain.chooseCC.setDisabled(False)
                        # enable correct subtab for P5
                        global erg_tabs
                        if P3.user_dimensions == 1:
                            erg_tabs.setTabEnabled(1, False)
                            erg_tabs.setTabEnabled(0, True)
                        elif P3.user_dimensions == 2:
                            erg_tabs.setTabEnabled(1, True)
                            erg_tabs.setTabEnabled(0, False)
                            
                        P3.button_dist.setDisabled(True)
                        P3.button_dist.setText('Distance Calculation Complete')
                        P3.progress1.setValue(100)
                        P3.button_eig.setDisabled(True)
                        P3.button_eig.setText('Embedding Complete')
                        P3.progress2.setValue(100)
                        P3.button_psi.setDisabled(True)
                        P3.button_psi.setText('Spectral Analysis Complete')
                        P3.progress3.setValue(100)
                        P3.button_nlsa.setDisabled(True)
                        P3.button_nlsa.setText('2D Movies Complete')
                        P3.progress4.setValue(100)

                        tabs.setTabEnabled(1, True)
                        tabs.setTabEnabled(2, True)
                        tabs.setTabEnabled(3, True)
                        noReturn()
                        tabs.setCurrentIndex(2)
                        P3.button_toP4.setDisabled(False)
                        print('\nUpdating eigenvectors tab...')
                        print('\trendering visualizations...')

                        # keep track of number of times user has re-embedded per PD:
                        P4.origEmbedFile = os.path.join(P1.user_directory, 'outputs_{}/topos/Euler_PrD/PrD_embeds.txt'.format(p.proj_name))
                        data = []
                        with open(P4.origEmbedFile) as values:
                            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                                data.append(column)
                        P4.origEmbed = []
                        idx = 0
                        for i in data[0]:
                            P4.origEmbed.append(int(i))
                            idx += 1
                        
                        P3.button_toP4.click()

                        # re-configure removals based on most recent user_removals.txt:
                        print('\tupdating user removals...')
                        fname = os.path.join(P1.user_directory,'outputs_{}/CC/user_removals.txt'.format(p.proj_name))
                        data = []
                        with open(fname) as values:
                            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                                data.append(column)
                        P4.trash_list = data[0]
                        p.trash_list = P4.trash_list

                        idx = 1 #PD index
                        for i in P4.trash_list:
                            if int(i) == int(1): #if PD set to True (remove)
                                P4.entry_PrD.setValue(idx)
                                P4.user_PrD = idx
                                P4.PrD_hist = idx
                                P4.trashAll[idx].setChecked(True)
                            idx += 1
                        P4.entry_PrD.setValue(1)

                        # re-configure anchors based on most recent user_anchors.txt:
                        print('\tupdating user anchors...')
                        if P3.user_dimensions == 1:
                            fname = os.path.join(P1.user_directory,'outputs_{}/CC/user_anchors.txt'.format(p.proj_name))
                            data = []
                            with open(fname) as values:
                                for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                                    data.append(column)
                            PrDs = data[0]
                            CC1s = data[1]
                            S1s = data[2]
                            Colors = data[3]

                            data_all = np.column_stack((PrDs,CC1s,S1s,Colors))
                            PrD = []
                            CC1 = []
                            S1 = []
                            Color = []
                            idx = 0
                            for i,j,k,l in data_all:
                                PrD.append(int(i))
                                CC1.append(int(j))
                                S1.append(int(k))
                                Color.append(int(l))
                                idx += 1

                            P4.anch_list = zip(PrD,CC1,S1,Color)
                            P4.anch_list, anch_zip = itertools.tee(P4.anch_list)
                            p.anch_list = list(anch_zip) #PrD,CC1,S1 for 1D; PrD,CC1,S1,CC2,S2 for 2D

                            idx = 0
                            for i in PrD:
                                P4.entry_PrD.setValue(int(i))
                                P4.user_PrD = i
                                P4.PrD_hist = i
                                if P4.trashAll[i].isChecked() == False: #avoids conflict
                                    P4.anchorsAll[i].setChecked(True)
                                P4.reactCoord1All[i].setValue(CC1[idx])
                                if S1[idx] == 1:
                                    P4.senses1All[i].setCurrentIndex(0)
                                elif S1[idx] == -1:
                                    P4.senses1All[i].setCurrentIndex(1)
                                idx += 1
                            P4.entry_PrD.setValue(1)

                        elif P3.user_dimensions == 2:
                            fname = os.path.join(P1.user_directory,'outputs_{}/CC/user_anchors.txt'.format(p.proj_name))
                            data = []
                            with open(fname) as values:
                                for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                                    data.append(column)
                            PrDs = data[0]
                            CC1s = data[1]
                            S1s = data[2]
                            CC2s = data[3]
                            S2s = data[4]
                            Colors = data[5]

                            data_all = np.column_stack((PrDs,CC1s,S1s,CC2s,S2s,Colors))
                            PrD = []
                            CC1 = []
                            S1 = []
                            CC2 = []
                            S2 = []
                            Color = []
                            idx = 0
                            for i,j,k,l,m,n in data_all:
                                PrD.append(int(i))
                                CC1.append(int(j))
                                S1.append(int(k))
                                CC2.append(int(l))
                                S2.append(int(m))
                                Color.append(int(n))
                                idx += 1
                            P4.entry_PrD.setValue(1)

                            P4.anch_list = zip(PrD,CC1,S1,CC2,S2,Color)
                            P4.anch_list, anch_zip = itertools.tee(P4.anch_list)
                            p.anch_list = list(anch_zip) #PrD,CC1,S1 for 1D; PrD,CC1,S1,CC2,S2 for 2D

                            idx = 0
                            for i in PrD:
                                P4.user_PrD = i
                                P4.PrD_hist = i
                                if P4.trashAll[i].isChecked() == False: #avoids conflict
                                    P4.anchorsAll[i].setChecked(True)
                                P4.reactCoord1All[i].setValue(CC1[idx])
                                P4.reactCoord2All[i].setValue(CC2[idx])
                                if S1[idx] == 1:
                                    P4.senses1All[i].setCurrentIndex(0)
                                elif S1[idx] == -1:
                                    P4.senses1All[i].setCurrentIndex(1)
                                if S2[idx] == 1:
                                    P4.senses2All[i].setCurrentIndex(0)
                                elif S2[idx] == -1:
                                    P4.senses2All[i].setCurrentIndex(1)
                                idx += 1

                        if p.resProj == 6: #anchor nodes entered, 'Compile Results` clicked
                            P4.btn_finOut.setText('  Recompile Results  ')
                            P4.recompute = 1
                            gotoP5(self)

                        elif p.resProj >= 7: #P5 onward
                            print('\nUpdating compilation tab...')

                            P4.btn_finOut.setText('  Recompile Results  ')
                            P4.recompute = 1
                            P5.entry_proc.setValue(p.ncpu)
                            P5.entry_opt.setDisabled(True)
                            if p.resProj == 7: #FindConformationalCoord.py complete
                                P5.progress5.setValue(100)
                                P5.button_CC.setText('Conformational Coordinates Complete')
                                P5.button_CC.setDisabled(True)
                                P5.button_erg.setDisabled(False)
                                gotoP5(self)
                            if p.resProj >= 8: #EL1D.py complete
                                print('\nUpdating energetics tab...')
                                P5.progress5.setValue(100)
                                P5.progress6.setValue(100)
                                P5.button_CC.setText('Conformational Coordinates Complete')
                                P5.button_CC.setDisabled(True)
                                P5.button_erg.setText('Energy Landscape Complete')
                                P5.button_erg.setDisabled(True)                                
                                P3.entry_proc.setDisabled(True)
                                P3.entry_psi.setDisabled(True)
                                P5.user_temperature = p.temperature
                                P5.entry_temp.setValue(p.temperature)
                                P5.entry_temp.setDisabled(True)
                                P5.button_toP6.setDisabled(True)
                                print('\ttemperature: %s' % (p.temperature))
                                if p.dim == 1:
                                    fnameOM = os.path.join(P1.user_directory,'outputs_{}/ELConc50/OM/S2_OM'.format(p.proj_name)) #occupancy
                                    fnameEL = os.path.join(P1.user_directory,'outputs_{}/ELConc50/OM/S2_EL'.format(p.proj_name)) #energy
                                    P4.Occ1d = np.fromfile(fnameOM, dtype=int)
                                    P4.Erg1d = np.fromfile(fnameEL)
                                    Erg1dMain.entry_width.setCurrentIndex(int(p.width_1D)-1)
                                    Erg1dMain.entry_width.model().item(int(p.width_1D)-1).setEnabled(False)
                                    Erg1dMain.plot_erg1d.update_figure() #updates 1d landscape plot
                                    Erg1dMain.button_traj.setDisabled(False)
                                elif p.dim == 2:
                                    #ZULU import file
                                    Erg2dMain.entry_width.setCurrentIndex(int(p.width_2D)-1) #check p.width_2D ZULU
                                    Erg2dMain.entry_width.model().item(int(p.width_2D)-1).setEnabled(False)
                                    Erg2dMain.plot_erg2d.update_figure() #updates 2d landscape plot
                                    Erg2dMain.btn_custComp.setDisabled(False)
                                    Erg2dMain.btn_leastComp.setDisabled(True)
                                tabs.setTabEnabled(4, True)
                                gotoP6(self)
                                
                            if p.resProj == 9: #PrepareOutputS2.py complete, temperature
                                if p.dim == 1:
                                    Erg1dMain.reprepare = 1
                                    Erg1dMain.progress7.setValue(100)
                                    Erg1dMain.button_traj.setText('Recompute 3D Trajectories')
                                    Erg1dMain.button_traj.setDisabled(False)
                                elif p.dim == 2: #ZULU: checkbox either Erg2dMain.btn_cust or Erg2dMain.btn_la
                                    Erg2dMain.reprepare = 1
                                    Erg2dMain.progress7.setValue(100)
                                    #if, then needed for below (ZULU)
                                    Erg2dMain.btn_custComp.setText('Recompute Custom Path')
                                    Erg2dMain.btn_leastComp.setText('Recompute LA Path')
                                    Erg2dMain.btn_custComp.setDisabled(False)
                                    Erg2dMain.btn_leastComp.setDisabled(True)
                    
                    set_params.op(0) #send new GUI data to user parameters file
                    P4.user_PrD = 1
                    P4.PrD_hist = 1
                    P4.entry_PrD.setValue(1)
                    if p.resProj >= 5:
                        P4.viz2.update_euler()

                    MainWindow.restoreComplete = 1 #previous project completely restored
                    print('\nLoading complete.')

                except IndexError:
                    if userInput == 'empty':
                        p.resProj = 0
                        box = QtGui.QMessageBox(self)
                        box.setWindowTitle('%s Error' % progname)
                        box.setText('<b>Input Error</b>')
                        box.setIcon(QtGui.QMessageBox.Warning)
                        box.setFont(font_standard)
                        box.setInformativeText('Incorrect file structure detected.\
                                                <br /><br />\
                                                Restarting ManifoldEM.')
                        box.setStandardButtons(QtGui.QMessageBox.Ok)
                        box.setDefaultButton(QtGui.QMessageBox.Ok)
                        ret = box.exec_()

                        # restart program:
                        try:
                            getProcess = psutil.Process(os.getpid())
                            for handler in getProcess.open_files() + getProcess.connections():
                                os.close(handler.fd)
                        except Exception as e:
                            logging.error(e)

                        python = sys.executable
                        os.execl(python, python, * sys.argv)

                    else: #prevents infinite error-loop if erroneous user_inputs.txt entered via CLI arg: --input
                        box = QtGui.QMessageBox(self)
                        box.setWindowTitle('%s Error' % progname)
                        box.setText('<b>Input Error</b>')
                        box.setIcon(QtGui.QMessageBox.Warning)
                        box.setFont(font_standard)
                        box.setInformativeText('Incorrect file structure detected.\
                                                <br /><br />\
                                                Exiting ManifoldEM.')
                        box.setStandardButtons(QtGui.QMessageBox.Ok)
                        box.setDefaultButton(QtGui.QMessageBox.Ok)
                        ret = box.exec_()

                        # force quit program:
                        app = QApplication(sys.argv)
                        app.closeAllWindows()

            else:
                p.resProj = 0
                box = QtGui.QMessageBox(self)
                box.setWindowTitle('%s Error' % progname)
                box.setText('<b>Input Error</b>')
                box.setIcon(QtGui.QMessageBox.Warning)
                box.setFont(font_standard)
                box.setInformativeText('No files were selected.\
                                        <br /><br />\
                                        Exiting ManifoldEM.')
                box.setStandardButtons(QtGui.QMessageBox.Ok)
                box.setDefaultButton(QtGui.QMessageBox.Ok)
                ret = box.exec_()

                # force quit program:
                app = QApplication(sys.argv)
                app.closeAllWindows()

        else:
            p.resProj = 0
            pass

    # End of resume progress
    # =========================================================================


    def onTabChange(self,i):
        if i == 1: #if changed to tab2
            if P2.noReturnFinal == False: #if user hasn't already `finalized` all params via P2
                # =============================================================
                # Force update to p.py if parameters weren't toggled:
                # =============================================================
                p.avg_vol_file = P1.user_volume
                p.img_stack_file = P1.user_stack
                p.align_param_file = P1.user_alignment
                p.mask_vol_file = P1.user_mask
                p.proj_name = P1.user_name
                p.pix_size = P1.user_pixel
                p.obj_diam = P1.user_diameter
                p.resol_est = P1.user_resolution
                p.ap_index = P1.user_aperture
                p.sh = P1.user_shannon
                p.ang_width = P1.user_width
                p.S2rescale = P1.S2rescale
                p.relion_data = P1.relion_data
                set_params.op(0) #send new GUI data to user parameters file
                # =============================================================
                Data.op(P1.user_alignment)
                time.sleep(1) #wait time to ensure files are completely written
                fname = open(os.path.join(P1.user_directory,'outputs_{}/selecGCs'.format(p.proj_name)), 'rb')
                data = pickle.load(fname)
                P1.S2 = data['S2']
                P1.CG = data['CG']
                fname.close()

                if np.shape(P1.CG)[0] <= 2: #if less than 2, number of jobs = 0 or 1; won't work
                    msg = 'The most populated bins do not hold enough data points\
                            to meet the minimum threshold criteria.\
                            <br /><br />\
                            Please recheck all parameters.\
                            More experimental data may be required to proceed.'
                            
                    box = QtGui.QMessageBox(self)
                    box.setWindowTitle('%s Error' % progname)
                    box.setText('<b>Input Error</b>')
                    box.setFont(font_standard)
                    box.setIcon(QtGui.QMessageBox.Warning)
                    box.setInformativeText(msg)
                    box.setStandardButtons(QtGui.QMessageBox.Ok)        
                    ret = box.exec_()
                    # return user to page 1:
                    tabs.setCurrentIndex(0)
                    tabs.setTabEnabled(1, False)
                    
                else:
                    # =========================================================
                    # Calculate PrD occupancies for figure on P2:
                    # =========================================================
                    fname = open(os.path.join(P1.user_directory,'outputs_{}/selecGCs'.format(p.proj_name)), 'rb')
                    data = pickle.load(fname)
                    # all tessellated bins:
                    mid = np.floor(np.shape(data['CG1'])[0]/2)
                    NC1 = data['NC'][:int(mid)]
                    NC2 = data['NC'][int(mid):]
        
                    P1.all_PrDs = []
                    P1.all_occ = []
                    P1.thresh_PrDs = []
                    P1.thresh_occ = []
                    if len(NC1) >= len(NC2): #first half of S2
                        pd_all = 1
                        pd = 1
                        for i in NC1:
                            P1.all_PrDs.append(pd_all)
                            P1.all_occ.append(i)
                            if i >= p.PDsizeThL:
                                P1.thresh_PrDs.append(pd)
                                if i > p.PDsizeThH:
                                    P1.thresh_occ.append(p.PDsizeThH)
                                else:
                                    P1.thresh_occ.append(i)
                                pd += 1
                            pd_all += 1
                    else: #second half of S2
                        pd_all = 1
                        pd = 1
                        for i in NC2:
                            P1.all_PrDs.append(pd_all)
                            P1.all_occ.append(i)
                            if i >= p.PDsizeThL:
                                P1.thresh_PrDs.append(pd)
                                if i > p.PDsizeThH:
                                    P1.thresh_occ.append(p.PDsizeThH)
                                else:
                                    P1.thresh_occ.append(i)
                                pd += 1
                            pd_all += 1
        
                    # ad hoc ratios to make sure S2 volume doesn't freeze-out due to too many particles to plot:
                    ratio1 = float(sum(P1.all_occ))/100
                    ratio2 = float(sum(P1.all_occ))/5000
                    MainWindow.S2_rho = min(P1.S2_density_all, key=lambda x:abs(x-ratio1)) #find value in list is closest to ratio1
                    P1.S2_density_all = list(filter(lambda a: a < int(sum(P1.all_occ)), P1.S2_density_all))
                    P1.S2_density_all = list(filter(lambda a: a > int(ratio2), P1.S2_density_all))
                    P2.viz1.update_S2_density_all()
                    P2.viz1.update_S2_params()
        
                    # draw/update plots on Distribution tab:
                    P2.viz1.update_scene2()
                    P2.viz1.update_scene1()
        
                    # read points from tessellated sphere:
                    PrD_map1 = os.path.join(P1.user_directory,'outputs_{}/topos/Euler_PrD/PrD_map1.txt'.format(p.proj_name))
                    PrD_map1_eul = []
                    with open(PrD_map1) as values:
                        for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                            PrD_map1_eul.append(column)
        
                    P1.all_phi = PrD_map1_eul[2]
                    P1.all_theta = PrD_map1_eul[1]
        
                    fname.close()
                    # =========================================================
        
        if i == 3: #auto-update Mayavi_Rho figure
            # only update figures (time consuming) if S2_scale and S2_iso have changed:
            if (MainWindow.S2_scale != MainWindow.S2_scale_prev) or (MainWindow.S2_iso != MainWindow.S2_iso_prev) or (MainWindow.restoreComplete == 0):
                P3.dictGen = False
                P4.viz2.update_S2()
                P4.viz2.update_scene3()
                P3.dictGen = True
                MainWindow.S2_scale_prev = MainWindow.S2_scale
                MainWindow.S2_iso_prev = MainWindow.S2_iso
                p.S2rescale = MainWindow.S2_scale
                p.S2iso = MainWindow.S2_iso
                set_params.op(0) #send new GUI data to user parameters file
            if P3.user_dimensions == 1:
                P4.anchorsAll[P4.user_PrD].setDisabled(False)

    def closeEvent(self, ce): #safety message if user clicks to exit via window button
        if P3.dictGen is False:
            msg = 'Performing this action will close the program.\
                   Current progress can be resumed at a later time\
                   via the <i>params</i> folder.\
                   <br /><br />\
                   Do you want to proceed?'
        else:
            msg = 'Performing this action will close the program.\
                   <br /><br />\
                   If you have made any PD assignments (anchors or removals)\
                   that have not yet been saved, please do so first\
                   via the <i>PD Selections</i> subwindow on the\
                   <i>Eigenvectors</i> tab.\
                   <br /><br />\
                   All other progress can be resumed at a later time\
                   via the <i>params</i> folder.\
                   <br /><br />\
                   Do you want to proceed?'

        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Warning' % progname)
        box.setText('<b>Exit Warning</b>')
        box.setFont(font_standard)
        box.setIcon(QtGui.QMessageBox.Warning)
        box.setInformativeText(msg)
        box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
        reply = box.exec_()
        if reply == QtGui.QMessageBox.Yes:
            #self.close()
            app = QtGui.QApplication.instance()
            app.closeAllWindows()
        else:
            ce.ignore()

    def fileRestart(self):
        if P3.dictGen is False:
            msg = 'Performing this action will restart the program.\
                   Upon reloading, current progress can be resumed\
                   via the <i>params.pkl</i> folder.\
                   <br /><br />\
                   Do you want to proceed?'
        else:
            msg = 'Performing this action will restart the program.\
                   <br /><br />\
                   If you have made any PD assignments (anchors or removals)\
                   that have not yet been saved, please do so first\
                   via the <i>PD Selections</i> subwindow on the\
                   <i>Eigenvectors</i> tab.\
                   <br /><br />\
                   All other progress can be resumed upon reloading\
                   via the <i>params.pkl</i> folder.\
                   <br /><br />\
                   Do you want to proceed?'
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Warning' % progname)
        box.setText('<b>Restart Warning</b>')
        box.setFont(font_standard)
        box.setIcon(QtGui.QMessageBox.Warning)
        box.setInformativeText(msg)
        box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
        reply = box.exec_()
        if reply == QtGui.QMessageBox.Yes:
            try:
                p = psutil.Process(os.getpid())
                for handler in p.open_files() + p.connections():
                    os.close(handler.fd)
            except Exception as e:
                logging.error(e)

            python = sys.executable
            os.execl(python, python, * sys.argv)
        else:
            pass

    def guide_import(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('')
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>Import Tab</b>')
        box.setFont(font_standard)
        box.setInformativeText("<span style='font-weight:normal;'>\
                                Input relevant experimental data into the\
                                corresponding entries. Once all entries have been filled,\
                                click the <i>View Orientation Distribution</i> button\
                                to proceed to the next tab.\
                                <br /><br />\
                                <b>Average Volume:</b>\
                                the Coulomb potential map obtained from 3D classification.\
                                <dl>\
                                    <dd>Accepted format: <i>.mrc</i></dd>\
                                    <dd> </dd>\
                                </dl>\
                                <br />\
                                <b>Alignment File:</b>\
                                the parameter file linking the 2D particle images to their\
                                corresponding microscopy attributes.\
                                <dl>\
                                    <dd>Accepted format: <i>.star</i></dd>\
                                    <br />\
                                    <dd>Requried parameters:\
                                    <i>Image Name, Angle Rot, Angle Tilt, Angle Psi,\
                                    OriginX, OriginY, Defocus U, Defocus V, Voltage,\
                                    Spherical Aberration, Amplitude Contrast</i</dd>\
                                    <dd> </dd>\
                                </dl>\
                                <br />\
                                <b>Image Stack:</b>\
                                the stack of individual 2D particle images obtained from \
                                micrograph segmentation.\
                                <dl>\
                                    <dd>Accepted format: <i>.mrcs</i></dd>\
                                    <dd> </dd>\
                                </dl>\
                                <br />\
                                <b>Mask Volume:</b>\
                                a volumetric mask whereby all exterior voxels/pixels will be ignored (optional).\
                                If no mask is supplied, an annular mask will be applied.\
                                <dl>\
                                    <dd>Accepted format: <i>.mrc</i></dd>\
                                    <dd> </dd>\
                                </dl>\
                                <p><b>Project Name:</b>\
                                the name of the project for which files will be created containing all program\
                                outputs (optional).\
                                If no path is supplied, the current date and time will be used.\
                                <br /><br />\
                                <b>Pixel Size:</b>\
                                the pixel size of the cameras used in experiment (in Angstroms).\
                                <br /><br />\
                                <b>Object Diameter:</b>\
                                the maximum width of the known protein structure; if unknown,\
                                the maximum width of the particle across all obtained projection\
                                directions.\
                                <br /><br />\
                                <b>Resolution:</b>\
                                the resolvability in the Coulomb potential map of\
                                the particle obtained from Fourier Shell Correlation.\
                                <br /><br />\
                                <b>Aperture Index:</b>\
                                integer describing the aperture size in terms of the Shannon angle.\
                                <br /><br />\
                                <b>Shannon Angle:</b>\
                                used to calculate the orientation bin size.\
                                <dl>\
                                    <dd>Defined via:\
                                    <i>Shannon angle</i> = <i>resolution</i> / <i>object diameter</i></dd>\
                                    <dd> </dd>\
                                <\dl>\
                                <br />\
                                <b>Angle Width:</b>\
                                the width of the aperture (in radians); cannot exceed sqrt(4pi).\
                                <dl>\
                                    <dd>Defined via:\
                                    <i>angle width</i> = <i>Shannon angle</i> * <i>aperture index</i>\
                                    </dd>\
                                    <dd> </dd>\
                                <\dl>\
                                <br /><br /></p></span>")
        box.setStandardButtons(QtGui.QMessageBox.Ok)        
        ret = box.exec_()

    def guide_orientations(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('')
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>Distribution Tab</b>')
        box.setFont(font_standard)
        box.setInformativeText("<span style='font-weight:normal;'>\
                                View the particle's <i>S2 Orientation Distribution</i>\
                                in tandem with its corresponding\
                                <i>Electrostatic Potential Map</i>. Both figures are\
                                synced such that navigating the\
                                camera within either figure will automatically update\
                                the view of the other.\
                                <br /><br />\
                                <b>S2 Orientation Distribution:</b>\
                                the occupancy of projections as mapped\
                                across the S2 angular space, with corresponding heatmap\
                                defining the relative spatial density of projections.\
                                <br /><br />\
                                <b>Electrostatic Potential Map:</b>\
                                the reconstructed volume obtained from 3D classification, with\
                                contour depth modulated via the <i>Isosurface</i>\
                                dropdown menu.\
                                <br /><br />\
                                To alter the filtering for the minimum number of\
                                particles needed to compute a given projection direction,\
                                click the <i>PD Thresholding</i> button.                              "
                               "<br /><br />\
                                When ready, click the <i>Bin Particles</i> button\
                                to proceed to the next tab.\
                                </span>")
        box.setStandardButtons(QtGui.QMessageBox.Ok)        
        ret = box.exec_()

    def guide_embedding(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>Embedding Tab</b>')
        box.setFont(font_standard)
        box.setInformativeText("<span style='font-weight:normal;'>\
                                The <i>Processors</i> should be set\
                                before pressing the <i>Distances</i> button,\
                                and can not be changed again until all processes are complete.\
                                <br /><br />\
                                The <i>Eigenvectors</i> value determines the number of leading\
                                Diffusion Map eigenvectors to process during NLSA,\
                                and cannot be changed once <i>Spectral Analysis</i> has been initiated.\
                                <br /><br />\
                                The <i>Dimensions</i> can be altered\
                                at any time on subsequent\
                                tabs, and represents the number of conformational coordinates incorporated\
                                when generating the energy landscape. This feature is restricted to 1D\
                                for the current (Beta) release.\
                                <br /><br />\
                                Once the distance calculations, embedding, spectral analysis, and 2D movies have\
                                been computed via the <i>Distances</i>, <i>Embedding</i>,\
                                <i>Spectral Analysis</i>, and <i>Compile 2D Movies</i> buttons,\
                                the user can navigate to the <i>Eigenvectors</i> tab via the\
                                <i>View Eigenvectors</i> button at the bottom of the screen.\
                                </span>")
        box.setStandardButtons(QtGui.QMessageBox.Ok)        
        ret = box.exec_()

    def guide_eigenvectors(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>Eigenvectors Tab</b>')
        box.setFont(font_standard)
        box.setInformativeText("<span style='font-weight:normal;'>\
                                <b>Selecting PDs:</b>\
                                Each node encircling the <i>Electrostatic Potential Map</i>\
                                represents a unique projection direction (PD) taken\
                                from that particle's thresholded <i>S2 Orientation Distribution</i>.\
                                Rotate the volume with the mouse (or via the arrows on the\
                                <i>PD</i> box below) to view its corresponding characteristics\
                                from each PD, including that PD's set of unique eigenvectors\
                                (Psi 1-8), <i>2D Class Average</i>, and <i>Eigenvalue Spectrum</i>.\
                                <br /><br />\
                                <b>PD Attributes:</b>\
                                Within a single PD, the properties of each eigenfuction can be explored\
                                by clicking the corresponding <i>View Psi</i> button underneath its image.\
                                For each eigenvector (Psi), a new window will open with a\
                                subtab entitled <i>Movie Player</i>, along with other tabs for analysis of the embeddings.\
                                If visual glitches are seen while viewing the image sequences\
                                within the <i>Movie Player</i> tab, the\
                                <i>2D Embedding</i> tab can be used to encircle the deviant clusters\
                                related to those behaviors and remove them (done via mouse clicks on the\
                                canvas, followed by the <i>Remove Cluster</i> button). The coordinates for the\
                                <i>2D Embedding</i> plot will reflect those chosen on the <i>3D Embedding</i> tab.\
                                After a cluster has been removed, the decision can be finalized via the\
                                <i>Update Manifold</i> button, which re-embeds the manifold with these\
                                new revisions in place.\
                                Once re-embedding occurs, it is also possible to revert the manifold back to\
                                its original embedding via the <i>Revert Manifold</i> button.\
                                <br /><br />\
                                <b>Selecting Anchors:</b>\
                                Anchor nodes must be set to optimize performance of the\
                                <i>Belief Propagation</i> algorithm - in which the eigenvectors are matched\
                                across all PDs and assigned their appropriate <i>Senses</i>. <i>Sense</i>\
                                defines the directionality of the movie; i.e., whether the image sequence\
                                for that anchor is playing forward (FWD) or in reverse (REV);\
                                with directionality arbitrary defined so long as it remains consistent across\
                                all PDs. These anchors and their corresponding senses can be set in the\
                                <i>Set PD Anchors</i> box - making sure that only eigenvectors of highest\
                                user certainty are chosen.\
                                As a contextual example, one such anchor decision might follow for ribosomal data:\
                                <br /><br />\
                                <i>Given a highly informative PD, which Psi (1-8) represents\
                                the readily-identifiable ratcheting motion of the ribosome,\
                                and is this motion seen as playing FWD or REV for that Psi?\
                                Next, is there another (preferably distant) PD where this same\
                                motion can also be easily identified across its set of eigenvectors? (etc.) </i>\
                                <br /><br />\
                                Upon confirmation of an anchor, the colored node for that PD will turn white\
                                in the accompanying figure. The original colors of the nodes (blue, red, etc.) are\
                                assigned based on spatial affinity, whereby different colored islands are separated by\
                                some distance considered too far for information to accurately pass between. As such,\
                                informative anchors should be set by the user in each of the colored islands present.\
                                <br /><br />\
                                The parameters for all selected anchors can be viewed at\
                                once via the <i>Review Anchors</i> button. As a note, if more than one\
                                dimension was chosen on the preceeding tab, anchors for two distinct\
                                eigenvectors must instead be chosen for a set of highest certainty PDs.\
                                Upon confirming the recommended number of anchor nodes, final outputs\
                                can be generated via the <i>Compile Results</i> button.\
                                </span>")
        box.setStandardButtons(QtGui.QMessageBox.Ok)        
        ret = box.exec_()

    def guide_outputs(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>Compilation Tab</b>')
        box.setFont(font_standard)
        box.setInformativeText("<span style='font-weight:normal;'>\
                                The <i>Number of Processors</i> can be set again for convenience\
                                before proceeding with final outputs.\
                                <br /><br />\
                                The pre-quenching temperature defines the temperature of the experiment\
                                before rapid freezing, and can be set via the <i>Temperature</i>\
                                controls.\
                                <br /><br />\
                                The user can choose to save the optical flow images obtained during\
                                computation of Conformational Coordinates via <i>Export Optical Flow</i>.\
                                <br /><br />\
                                Once conformational coordinates, energy landscape, and 3D Trajectories have\
                                been computed via the <i>Find Conformational Coordinates</i>,\
                                <i>Energy Landscape</i>, and <i>Compute 3D Trajectories</i>\
                                buttons, the user can navigate to the <i>Energetics</i> tab via the\
                                <i>View Energy Landscape</i> button.\
                                </span>")
        box.setStandardButtons(QtGui.QMessageBox.Ok)        
        ret = box.exec_()

    def guide_dyn_1d(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>1D Energetics Tab</b>')
        box.setFont(font_standard)
        box.setInformativeText("<span style='font-weight:normal;'>\
            The <i>1D Energy Path</i> is plotted along the\
            1D conformational coordinate. The energy along these coordinates is taken\
            from the corresponding occupancies via the Boltzmann factor.\
            The particle occupancies along this same trajectory can be plotted\
            via the <i>Occupancy Map</i>.\
            </span>")
        box.setStandardButtons(QtGui.QMessageBox.Ok)        
        ret = box.exec_()

    def guide_dyn_custom(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>2D Energetics Tab</b>')
        box.setFont(font_standard)
        box.setInformativeText('Custom data points can be selected by clicking directly on the energy landscape.\
            <br /><br />\
            As points are chosen, a series of linear paths will be created between subsequent selections. \
            The combined energy of all selected path segments on the energy landscape is automatically \
            displayed to the right of the plot in the <i>Custom path integral</i> box. <br /> <br /> \
            \
            Set the <i>Path width</i> parameter to modulate the area of information included \
            within the neighborhood of the path. Path widths are calculated via an incorporation \
            of pixels within the neighborhood of each selected point.<br /> <br /> \
            \
            Press <i>Reset points</i> to reset the plot and erase all points, paths and integrated\
            energies.<br /> <br /> \
            \
            After selecting a path, the 3d reconstruction along the trajectory can be created by\
            clicking <i>Compute 3D Trajectories</i> within the <i>Custom Path Analysis</i> section.')
        box.setStandardButtons(QtGui.QMessageBox.Ok)        
        ret = box.exec_()

    def guide_dyn_action(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>2D Energetics Tab</b>')
        box.setFont(font_standard)
        box.setInformativeText('To use this feature, a path of least action must first be precomputed\
            using an external program (e.g., see <i>POLARIS: Path of Least Action Analysis on Energy Landscapes</i>, Seitz and Frank).\
            <br /><br />\
            Once you have obtained a path of least action, import its coordinates into this GUI via the <i>Import LA Path</i> button.\
            <br /> <br /> \
            Finally, the 3d reconstruction along this LA trajectory can be created by clicking <i>Compute 3D Trajectories</i>\
            within the <i>Least Action Path Analysis</i> section.')
        box.setStandardButtons(QtGui.QMessageBox.Ok)        
        ret = box.exec_()

    def cameraInfo(self): #camera handling guide
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s Help' % progname)
        box.setText('<b>3D Visualizations</b>')
        box.resize(800,400)
        box.setFont(font_standard)
        box.setInformativeText('<b>Rotating:</b> place the mouse pointer on top of the visualization window, and keep the left\
                                mouse button pressed while dragging in the appropriate direction.\
                                An in-plane rotation can also be performed by rotating as above while\
                                holding down the "Control" button.<br /><br />\
                                \
                                <b>Zooming:</b> to zoom in and out of the scene, place the mouse pointer inside the visualization window, and\
                                keep the right mouse button pressed while dragging the mouse upwards or downwards. If available,\
                                the mouse scroll wheel can also be used.<br /><br />\
                                \
                                <b>Panning:</b> to translate the center of the scene, keep the left mouse button pressed\
                                in combination with holding down the <i>Shift</i> key while dragging the mouse in the appropriate direction.\
                                If available, the middle mouse button can also be pressed while dragging the mouse to pan in the appropriate\
                                direction.<br /><br />\
                                \
                                <b>Keyboard Interactions:</b> If problems arise between scene navigation and cursor movement,\
                                a nonstandard key may have been pressed. Pressing <b>t</b> (trackball mode) may reset this behavior.\
                                For more information on using Mayavi, including other keyboard shortcuts, please visit:<br /><br />\
                                http://docs.enthought.com/mayavi/mayavi/application.html'
                                )
        box.setStandardButtons(QtGui.QMessageBox.Ok)
        box.setDefaultButton(QtGui.QMessageBox.Ok)
        ret = box.exec_()

    def about(self):
        box = QtGui.QMessageBox(self)
        box.setWindowTitle('%s About' % progname)
        box.setText('<b>%s v.%s Python 3</b>' % (progname, progversion))
        box.setFont(font_standard)
        iconDir = os.path.join(pyDir, 'icons/200x200.png')
        if iconDir:
            box.setIconPixmap(QtGui.QPixmap(iconDir)) #ZULU
        box.setInformativeText('<span style="font-weight:normal;">\
                                %s Frank Lab, 2021\
                                <br /><br />\
                                <b>LICENSE:</b>\
                                <br /><br />\
                                You should have received a copy of the GNU General Public\
                                License along with this program.\
                                <br /><br />\
                                <b>CONTACT:</b>\
                                <br /><br />\
                                Please refer to our user manual for all inquiries, including\
                                preferred methods for contacting our team for software support.\
                                <br /><br />\
                                </span>' % (u"\u00A9"))
        box.setStandardButtons(QtGui.QMessageBox.Ok)        
        ret = box.exec_()
        
    def gotoP1(self):
        tabs.setCurrentIndex(0)

    def gotoP2(self):
        P1.button_toP2.setDisabled(True)
        p.S2iso = MainWindow.S2_iso
        p.S2iso_prev = MainWindow.S2_iso

        if (P1.user_volume != '' and P1.user_stack != '' and P1.user_alignment != '' and
            P1.user_shannon != 0.0 and P1.user_width != 0.0):
            if os.path.exists(os.path.join(P1.user_directory, 'outputs_{}'.format(p.proj_name))):
                box = QtGui.QMessageBox(self)

                box.setWindowTitle('%s Conflict' % progname)
                box.setText('<b>Directory Conflict</b>')
                box.setFont(font_standard)
                box.setIcon(QtGui.QMessageBox.Warning)
                box.setInformativeText('The directory <i>outputs_{}</i> already exists, from either a previous project\
                                        or from re-adjusting the hyper-parameters within the current run.<br /><br />\
                                        \
                                        Do you want to overwrite this data and proceed?'.format(p.proj_name))
                box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
                reply = box.exec_()

                if reply == QtGui.QMessageBox.Yes: #overwrite directory and continue to page 2
                    call(["rm", "-r", os.path.join(P1.user_directory, 'outputs_{}'.format(p.proj_name))]) #delete output directory if it exists
                    if os.path.exists(os.path.join(pyDir, 'params_{}.pkl'.format(p.proj_name))):
                        os.remove(os.path.join(pyDir, 'params_{}.pkl'.format(p.proj_name)))
                    set_params.op(0) #send new GUI data to user parameters file
                    p.create_dir()
                    # proceed to page 2:
                    tabs.setTabEnabled(1, True)
                    tabs.setCurrentIndex(1)

                elif reply == QtGui.QMessageBox.No:
                    pass

            elif os.path.exists(os.path.join(pyDir, 'params_{}.pkl'.format(p.proj_name))):
                box = QtGui.QMessageBox(self)

                box.setWindowTitle('%s Conflict' % progname)
                box.setText('<b>Parameters Conflict</b>')
                box.setFont(font_standard)
                box.setIcon(QtGui.QMessageBox.Warning)
                box.setInformativeText('A previous <i>params_{}.pkl</i> file already exists within the initial project directory.\
                                        <br /><br />\
                                        Do you want to overwrite this data and proceed?'.format(p.proj_name))
                box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
                reply = box.exec_()

                if reply == QtGui.QMessageBox.Yes: #overwrite directory and continue to page 2
                    os.remove(os.path.join(pyDir, 'params_{}.pkl'.format(p.proj_name)))
                    set_params.op(0) #send new GUI data to user parameters file
                    p.create_dir()
                    # proceed to page 2:
                    tabs.setTabEnabled(1, True)
                    tabs.setCurrentIndex(1)

                elif reply == QtGui.QMessageBox.No:
                    pass

            else:
                set_params.op(0) #send new GUI data to user parameters file
                p.create_dir()
                # proceed to page 2:
                tabs.setTabEnabled(1, True)
                tabs.setCurrentIndex(1)

        else:
            box = QtGui.QMessageBox(self)
            box.setWindowTitle('%s Error' % progname)
            box.setText('<b>Input Error</b>')
            box.setFont(font_standard)
            box.setIcon(QtGui.QMessageBox.Information)
            box.setInformativeText('All values must be complete and nonzero.')
            box.setStandardButtons(QtGui.QMessageBox.Ok)
            box.setDefaultButton(QtGui.QMessageBox.Ok)
            ret = box.exec_()

        P1.button_toP2.setDisabled(False)


    def gotoP3(self):
        msg = 'Performing this action will lock in all user inputs up to this point.<br /><br />\
                \
                Do you want to proceed?'
        box = QtGui.QMessageBox(self)

        box.setWindowTitle('%s Warning' % progname)
        box.setText('<b>Progress Warning</b>')
        box.setFont(font_standard)
        box.setIcon(QtGui.QMessageBox.Warning)
        box.setInformativeText(msg)
        box.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
        reply = box.exec_()

        if reply == QtGui.QMessageBox.Yes:
            # disable page 1:
            noReturn()
            p.resProj = 1
            set_params.op(0) #send new GUI data to user parameters file
            # continue to page 3:
            tabs.setCurrentIndex(2)
            tabs.setTabEnabled(2, True)
        else:
            pass

    global gotoP4
    def gotoP4(self):
    # continue to page 4:
        tabs.setCurrentIndex(3)
        tabs.setTabEnabled(3, True)

    global gotoP5
    def gotoP5(self):
    # continue to page 5:
        tabs.setCurrentIndex(4)
        tabs.setTabEnabled(4, True)

    global gotoP6
    def gotoP6(self):
    # continue to page 6:
        P5.button_toP6.setDisabled(True)
        tabs.setCurrentIndex(5)
        tabs.setTabEnabled(5, True)

if __name__ == '__main__':
    print('# =============================')
    print('#  %s' % (progname))
    print('#  v.%s Python 3' % (progversion))
    print('# =============================')
    print('')

    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])

    app.setStyle('fusion') #default style across all OS

    # =========================================================================
    # menu bar app name not working on macOS;
    # see: <https://stackoverflow.com/questions/7827430
    # ... /setting-mac-osx-application-menu-menu-bar-item-to
    # ... -other-than-python-in-my-pyth>
    QtCore.QCoreApplication.setApplicationName(progname) #if non-macOS menu bar
    # =========================================================================
        
    # set app icon for tray:
    iconDir = os.path.join(pyDir, 'icons')
    app_icon = QtGui.QIcon()
    app_icon.addFile(os.path.join(iconDir, '256x256.png'), QtCore.QSize(256,256))
    app.setWindowIcon(app_icon)

    w = MainWindow()
    w.setWindowTitle(progname)
    sys.exit(app.exec_())