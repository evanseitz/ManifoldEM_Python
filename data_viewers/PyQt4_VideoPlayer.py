import sys, os
from pyface.qt import QtGui, QtCore
os.environ['ETS_TOOLKIT'] = 'qt4'
from scipy import misc
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore', '.*Using default event loop.*')

#################################################################################
# STANDALONE VIDEO PLAYER (PyQt4)
#################################################################################
# HOW TO USE:
## 1. make sure all images are in the `frames` folder within the same directory as the python script:
pyDir = os.path.dirname(os.path.abspath(__file__)) #python file location
imgDir = os.path.join(pyDir, 'frames')
## 2. ... and that all images are in correct format (default .png, as seen below)
## 3. ... and in the correct ordering (00, 01, 02, ...)
#################################################################################
# Copyright (c) Columbia University Evan Seitz 2018-2020
#################################################################################

class VidCanvas(QtGui.QDialog):
    imgDir = ''
    img_paths = []
    imgs = []
    frames = 0 #total number of frames
    run = 0 #switch, {-1,0,1} :: {backwards,pause,forward}
    f = 0 #frame index (current frame)
    rec = 0 #safeguard for recursion limit
    speed = .05 #playback speed, can't be too fast on linux (see below)

    def __init__(self, parent=None):
        super(VidCanvas, self).__init__(parent)

        i = 0
        for root, dirs, files in os.walk(imgDir):
            for file in sorted(files):
                if not file.startswith('.'): #ignore hidden files
                    if file.endswith(".png"):
                        VidCanvas.img_paths.append(os.path.join(root, file))
                        VidCanvas.imgs.append(misc.imread(VidCanvas.img_paths[i]))
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
        self.toolbar = NavigationToolbar(VidCanvas.canvas, self)
        self.currentIMG = self.ax.imshow(VidCanvas.imgs[0])  #plot initial data
        VidCanvas.canvas.draw() #refresh canvas

        # player control buttons:
        self.buttonF1 = QtGui.QPushButton(ur'\u29D0')
        self.buttonF1.clicked.connect(self.F1)
        self.buttonF1.setDisabled(False)
        self.buttonF1.setDefault(False)
        self.buttonF1.setAutoDefault(False)

        VidCanvas.buttonForward = QtGui.QPushButton(ur'\u25BB') #u25B6, u25BA
        VidCanvas.buttonForward.clicked.connect(self.forward)
        VidCanvas.buttonForward.setDisabled(False)
        VidCanvas.buttonForward.setDefault(True)
        VidCanvas.buttonForward.setAutoDefault(True)

        VidCanvas.buttonPause = QtGui.QPushButton(ur'\u25A1') #u25A0, u25FC
        VidCanvas.buttonPause.clicked.connect(self.pause)
        VidCanvas.buttonPause.setDisabled(True)
        VidCanvas.buttonPause.setDefault(False)
        VidCanvas.buttonPause.setAutoDefault(False)

        VidCanvas.buttonBackward = QtGui.QPushButton(ur'\u25C5') #u25C0, u25C4
        VidCanvas.buttonBackward.clicked.connect(self.backward)
        VidCanvas.buttonBackward.setDisabled(False)
        VidCanvas.buttonBackward.setDefault(False)
        VidCanvas.buttonBackward.setAutoDefault(False)

        self.buttonB1 = QtGui.QPushButton(ur'\u29CF')
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
        layout.addWidget(self.toolbar, 0,0,1,5)
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
        VidCanvas.canvas.start_event_loop(self.speed)

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


################################################################################
# overhead setup:

class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.stack = QtGui.QStackedWidget(self)
        vid = VidCanvas(self)
        self.stack.addWidget(vid)
        self.setCentralWidget(self.stack)
        self.show()


if __name__ == '__main__':

    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])

    w = MainWindow()
    sys.exit(app.exec_())
