######################################
# Author: 张庆儒
# SID: 516021910419
######################################

import sys
import time
import copy
import random
import numpy
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon,QPalette,QPixmap,QFont, QColor
from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from DrawNeuralNetwork import draw_neural_net
 

# Carve widget for displaying plot figure
class PlotCanvas(FigureCanvas):
 
    def __init__(self, parent=None, width=5, height=4, dpi=100, ax_num = 1):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        if ax_num != 0:
            self.axes = self.fig.add_subplot(111)
 
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    
    def clear_figure(self):
        plt.figure(self.fig.number)
        plt.clf()
 
    # plot neural network structure 
    def plot_nn_struc(self, nn_seq):
        print('Carve draw struc: %s'%str(nn_seq))
        self.axes = self.fig.add_subplot(111)
        draw_neural_net(self.axes, .1, .9, .01, 1., nn_seq)
        self.draw()

 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PlotCanvas()
    ex.show()
    sys.exit(app.exec_())