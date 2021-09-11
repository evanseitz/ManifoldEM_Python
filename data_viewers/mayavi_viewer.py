import sys, os
from pyface.qt import QtGui, QtCore
os.environ['ETS_TOOLKIT'] = 'qt4'
import os
import csv
import pandas as pd
import numpy as np
from traits.api import HasTraits,Instance,on_trait_change,Button
from traitsui.api import View,Item,HGroup,VGroup,Group
from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

#################################################################################
# ALIGNMENT FILE VIEWER (S2 EULER ANGLES)
#################################################################################
# HOW TO USE:
## load 'ManifoldEM' environment
## copy alignment file and remove its header
## change column numbers below to reflect angles in alignment file ('CHANGE 1', '2', '3')
## move copied file into same directory as this python script
#################################################################################
# Copyright (c) Columbia University Evan Seitz 2020
#################################################################################

class Mayavi_Scene(HasTraits):
    scene = Instance(MlabSceneModel, ())

    display_angle = Button('Display Euler Angles')

    @on_trait_change('display_angle')
    def view_anglesP2(self):
        viewS2 = self.scene.mlab.view(figure=Mayavi_Scene.fig1)
        azimuth = viewS2[0] #phi: 0-360
        elevation = viewS2[1] #theta: 0-180
        zoom = viewS2[2]
        print(azimuth, elevation)

    @on_trait_change('scene.activated')
    def update_scene(self):
        Mayavi_Scene.fig1 = mlab.figure(1, bgcolor=(.5,.5,.5))
        self.scene.mlab.clf(figure=Mayavi_Scene.fig1)

        pyDir = os.path.dirname(os.path.realpath(__file__)) #python file location
        relDir = os.path.join(pyDir, 'rel_angles.txt')

        rel_euler = []

        rel_euler = pd.read_csv(relDir, header=None, delim_whitespace=True)
        rel_theta = rel_euler[10] #CHANGE 1
        rel_phi = rel_euler[11] #CHANGE 2
        rel_psi = rel_euler[12] #CHANGE 3

        def sphere2cart(theta, phi):
            r = 10
            x = r*np.sin(phi)*np.cos(theta)
            y = r*np.sin(phi)*np.sin(theta)
            z = r*np.cos(phi)
            return x,y,z

        rel_X = []
        rel_Y = []
        rel_Z = []

        rel_theta_phi = np.column_stack((rel_theta,rel_phi))

        for i in rel_theta_phi:
            x,y,z = sphere2cart(float(i[0])*np.pi/180, float(i[1])*np.pi/180)
            rel_X.append(x)
            rel_Y.append(y)
            rel_Z.append(z)

        if 1:
            rel_plot = mlab.points3d(rel_X, rel_Y, rel_Z, scale_mode='none',
                                  scale_factor=.5, figure=Mayavi_Scene.fig1)
            rel_plot.actor.property.color = (0.0, 1.0, 1.0)

    view = View(VGroup(
                Group(
                Item('scene', editor = SceneEditor(scene_class=MayaviScene),
                    height=300, width=300, show_label=False),
                Item('display_angle',springy=True,show_label=False),
                ),
                ),
                resizable=True,
                )
                

class P1(QtGui.QWidget):
    def __init__(self, parent=None):
        super(P1, self).__init__(parent)
        layout = QtGui.QGridLayout(self)
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(10)

        self.viz1 = Mayavi_Scene()
        self.ui1 = self.viz1.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.ui1, 1, 1, 1, 9)


class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setGeometry(50, 50, 500, 500)   

        tab1 = P1(self)

        self.tabs = QtGui.QTabWidget(self)

        self.tabs.addTab(tab1, 'Alignment Viewer')

        self.groupscroll = QtGui.QHBoxLayout()
        self.groupscrollbox = QtGui.QGroupBox()

        self.MVB = QtGui.QVBoxLayout()
        self.MVB.addWidget(self.tabs)

        scroll = QtGui.QScrollArea()
        widget = QtGui.QWidget(self)
        widget.setLayout(QtGui.QHBoxLayout())
        widget.layout().addWidget(self.groupscrollbox)
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        self.groupscrollbox.setLayout(self.MVB)
        self.groupscroll.addWidget(scroll)
        self.setCentralWidget(scroll)
        self.show()

if __name__ == '__main__':
    app = QtGui.QApplication.instance()
    w = MainWindow()
    sys.exit(app.exec_())
