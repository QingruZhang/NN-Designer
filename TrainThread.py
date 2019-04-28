# -*- coding: utf-8 -*-
######################################
# Author: 张庆儒
# SID: 516021910419
######################################

import sys
import time, threading
import io
from PyQt5 import QtCore
from contextlib import redirect_stdout
from train_code.model_v3 import *
import matplotlib.pyplot as plt 


# Sub thread class for training NN
class TrainThread(QtCore.QThread):

    # update carve figure signal, pass to visualized_training instance 
    update_Carve_signal = QtCore.pyqtSignal(str)

    def __init__(self,content:dict, UI_stop_signal, epoch = 200, data_size=10000, \
            loss_fig = None, loss_ax = None, out_fig = None, out_ax = None, parent=None, *args, **kwargs):# UI_stop_signal = None, iface = conf.iface, parent = None, pkt_num = 0, filter_UI = None, *args, **kwargs):
        
        super(TrainThread, self).__init__(parent)
        self.data_type = content['dataset']
        self.optim_para = content['optim_para']
        self.in_dim = content['in_dim']
        self.out_dim = content['out_dim']
        self.epoch = epoch
        self.data_size = data_size
        self.layers = content['layers']

        # Generate data according to input choose
        self.fake_data = data_generating(self.data_type, self.data_size)
        self.model = Visualized_Trainer(self.optim_para, self.in_dim, self.out_dim, self.layers[1:], self.fake_data,
                                        loss_fig = loss_fig, out_fig = out_fig,
                                        update_Carve_signal = self.update_Carve_signal)
        self._join = self.model.join

        self.stop_sniff_signal = UI_stop_signal
        self.stop_sniff_signal.connect(self.join)

    # run thread function
    def run(self):
        print('Thread begin training')
        self.model.visualized_training(self.epoch)

    # join training function
    def join(self, isStop):
        self.model.join = True

