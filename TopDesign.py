# -*- coding: utf-8 -*-
import sys, time, copy
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon,QPalette,QPixmap,QFont, QColor
from PyQt5.QtWidgets import *
from SelectWidget import SelectWiget
from Carve import PlotCanvas
import TrainThread
from train_code import model_v3
import matplotlib.pyplot as plt 

######################################
# Author: 张庆儒
# SID: 516021910419
######################################


# =====================================
# Central Widget for top window
# =====================================
class CentralWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.init_UI()

    def init_UI(self):
        # Top UI layout, use grid layout design
        self.top_layout = QGridLayout()
        
        # Upper left widget, input the 
        self.select_widget = SelectWiget()
        self.select_widget.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))

        # Lower left widget, code display zone
        self.text_viewer = QTextBrowser(self)
        self.font = QFont("Ubuntu Mono", 10)
        self.font.setBold(True)
        self.init_text = '-------------Code Zone---------------'
        self.text_viewer.setText(self.init_text)
        self.text_viewer.setFont(self.font)

        # Carve widgets
        self.carve_loss = PlotCanvas(ax_num = 0)
        self.carve_out = PlotCanvas(ax_num = 0)

        # Add widget to central widget layout
        self.top_layout.addWidget(self.carve_out, 0,6,6,6)
        self.top_layout.addWidget(self.carve_loss, 6,6,6,6)
        self.top_layout.addWidget(self.select_widget,0,0,6,6)
        self.top_layout.addWidget(self.text_viewer, 6,0,6,6)

        # set central widget layout
        self.setLayout(self.top_layout)

    # Build parameter form input widget
    def get_content(self):
        content = self.select_widget.gene_content()
        return content

    # Clear all figure in central widget
    def clear_all_figure(self):
        self.carve_loss.clear_figure()
        self.carve_out.clear_figure()

    # Set text for text viewer
    def set_txtbroswer_str(self, text:str):
        self.text_viewer.setText(text)

    # Clear text broswer
    def clear_txtbroswer(self):
        self.text_viewer.setText(self.init_text)

    # Plot neural network structure
    def plot_structure(self, sep):
        self.carve_out.plot_nn_struc(sep)



# =====================================
# Top Window for GUI
# =====================================

class TopDesign(QMainWindow):

    # Stop trainning signal, pass to every training thread
    # to control their action
    stop_train_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.content = None
        self.epoch = 200
        self.data_size = 1000

        self.struc = True
        self.loss_fig = None
        self.loss_ax = None 
        self.out_fig = None

        self.initUI()

    def initUI(self):         
        # Set Top UI Logo
        self.setWindowIcon(QIcon('img/Logo.png'))    

        # Central Widget
        self.central_Widget = CentralWidget()
        self.loss_fig = self.central_Widget.carve_loss.fig
        self.loss_ax = None
        self.out_fig = self.central_Widget.carve_out.fig

        # self.menubar and toolbar
        self.statusbar = self.statusBar()
        self.statusbar.showMessage('Ready')

        # Menubar Action
        self.exitAct = QAction(QIcon('img/Exit.png'), '&Exit', self)        
        self.exitAct.setShortcut('Ctrl+Q')
        self.exitAct.setStatusTip('Exit application')
        self.exitAct.triggered.connect(self.close)

        self.viewStatAct = QAction('View statusbar', self, checkable=True)
        self.viewStatAct.setStatusTip('View statusbar')
        self.viewStatAct.setChecked(True)
        self.viewStatAct.triggered.connect(self.toggleMenu)


        # tool bar
        self.toolbar = self.addToolBar('Tool')

        # Generate code and structure plot button
        self.geneAct = QAction(QIcon('img/geneAct.png'), '&Visualize', self)
        self.geneAct.setShortcut('Ctrl+B')
        self.geneAct.setStatusTip('Generate Visulization')

        # Start training button, after generating code
        self.trainAct = QAction(QIcon('img/trainAct.png'), '&Training', self)
        self.trainAct.setShortcut('Ctrl+T')
        self.trainAct.setStatusTip('Start Training')

        # Stop training button, control subthread by self.stop_train_signal
        self.stoptrainAct = QAction(QIcon('img/stoptrainAct.png'), '&Stop Training', self)
        self.stoptrainAct.setShortcut('Ctrl+S')
        self.stoptrainAct.setStatusTip('Stop Training')

        # Show loss button
        self.showlossAct = QAction(QIcon('img/showlossAct.png'), '&Show Loss', self)
        self.showlossAct.setShortcut('Ctrl+Shift+B')
        self.showlossAct.setStatusTip('Show Training Process')

        # Show output button
        self.showoutputAct = QAction(QIcon('img/showoutputAct.png'), '&Show Output', self)
        self.showoutputAct.setShortcut('Ctrl+O')
        self.showoutputAct.setStatusTip('Show Output')

        # Clear all input and output button
        self.clearAct = QAction(QIcon('img/ClearAct.png'), '&Clear All', self)
        self.clearAct.setShortcut('Ctrl+Shift+C')
        self.clearAct.setStatusTip('Clear All Input and Output')

        # Add them to tool bar
        self.toolbar.addAction(self.geneAct)
        self.toolbar.addAction(self.trainAct)
        self.toolbar.addAction(self.stoptrainAct)
        self.toolbar.addAction(self.showlossAct)
        self.toolbar.addAction(self.showoutputAct)
        self.toolbar.addAction(self.clearAct)

        # Connect tool bar button signal to corsponding slot
        self.clearAct.triggered.connect(self.clear_all)
        self.stoptrainAct.triggered.connect(self.StopTrain)
        self.trainAct.triggered.connect(self.StartTrain)
        self.geneAct.triggered.connect(self.build_network)

        # meunbar
        self.menubar = self.menuBar()

        # Tool menu design
        self.toolMenu = self.menubar.addMenu('Tool(&T)')
        self.toolMenu.addAction(self.geneAct)
        self.toolMenu.addAction(self.trainAct)
        self.toolMenu.addAction(self.stoptrainAct)
        self.toolMenu.addAction(self.clearAct)

        # View menu design
        self.viewMenu = self.menubar.addMenu('View(&V)')
        self.viewMenu.addAction(self.showlossAct)
        self.viewMenu.addAction(self.showoutputAct)
        self.viewMenu.addAction(self.viewStatAct)

        # Quit menu
        self.quitMenu = self.menubar.addMenu('Quit(&Q)')
        self.quitMenu.addAction(self.exitAct)
        
        # Set central widget using a CentralWidget instance
        self.setCentralWidget(self.central_Widget)

        self.setGeometry(100, 50, 1550, 950)
        self.setWindowTitle('NN-Designer')  

    def toggleMenu(self, state):
        if state:
            self.statusbar.show()
        else:
            self.statusbar.hide()

    # Generate code and structure plot according to input parameters
    def build_network(self):
        self.content = self.central_Widget.get_content()
        print('Finish Building NN: %s'%str(self.content))
        self.code_str = model_v3.get_code(self.content['layers'])
        self.central_Widget.set_txtbroswer_str(self.code_str)

        if self.struc:
            print('Draw NN structure')
            self.unit_sep = []
            for layer in self.content['layers']:
                self.unit_sep.append(layer['in'])
            self.unit_sep.append(1)
            self.central_Widget.plot_structure(self.unit_sep)


    # Start a training thread to train generated neural network
    def StartTrain(self):
        print('Will Start Train with Content %s'%str(self.content))
        if self.content is not None:
            # Stop left thread first
            self.stop_train_signal.emit(True)
            # Start a subthread
            self.train_thread = TrainThread.TrainThread(self.content, self.stop_train_signal, epoch = self.epoch, data_size=self.data_size, 
                                        loss_fig = self.loss_fig, loss_ax = self.loss_ax, out_fig = self.out_fig, out_ax = None, )
            # Connect plot signal
            self.train_thread.update_Carve_signal.connect(self.draw_carve)
            # Start training
            self.train_thread.start()
            # Set buttons
            self.stoptrainAct.setEnabled(True)
            self.trainAct.setEnabled(False)

    # Stop every training threads
    def StopTrain(self):
        self.stop_train_signal.emit(True)
        self.stoptrainAct.setEnabled(False)
        self.trainAct.setEnabled(True)

    # Draw carve
    def draw_carve(self, carve:str):
        if carve == 'loss':
            self.central_Widget.carve_loss.draw()
        elif carve == 'out':
            self.central_Widget.carve_out.draw()

    # Clear function
    def clear_all(self):
        self.content = None
        self.central_Widget.select_widget.clear_layers()
        self.central_Widget.clear_all_figure()
        self.central_Widget.clear_txtbroswer()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    TopWindow = TopDesign()
    TopWindow.show()
    sys.exit(app.exec_())
    
