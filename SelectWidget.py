# -*- coding: utf-8 -*-
######################################
# Author: 张庆儒
# SID: 516021910419
######################################

import sys, time, copy
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon,QPalette,QPixmap,QFont, QColor
from PyQt5.QtWidgets import *
from DrawNeuralNetwork import draw_neural_net
import qtawesome, sip

def except_type(istr:str, otype:str, th, default):
	try:
		beta1 = eval(otype)(istr)
		beta1 = beta1 if beta1<th else default
	except:
		beta1 = default
	return beta1



# Top left widget for set NN parameters
# constract a dict and pass to Design function
class SelectWiget(QWidget):

	def __init__(self):
		super().__init__()
		# net type options
		self.net_types = [
			'Full Connect',
		]

		# Loss type options
		self.loss_types =[
			'MSE',
			'MAE',
		]

		# Dataset options
		self.DataSets = {
			'Mulitiply Square':[2,1],
			'Beale Function':[2,1],
			'Levi function N.13':[2,1],
			'Himmelblau function':[2,1],
		}
		self.data_names = list(self.DataSets.keys())

		# Optimizer options
		self.optim_names = [
			'SGD',
			'Adam',
			'Adagrad',
			'Adamax',
		]

		# Activation options
		self.Activations = [
			'relu',
			'sigmoid',
			'tanh',
		]

		# Initial content dict
		self.init_content = {
			'net_type':self.net_types[0],
			'dataset': self.data_names[0],
			'optim_para': { 'type':self.optim_names[0], 'param':{}, 'batch_size':10, 'loss':self.loss_types[0] },
			'layers':[
				{'id':0, 
				'in': self.DataSets[self.data_names[0]][0], 
				'out':None, 'bias':True, 'activation':'relu','loss':self.loss_types[0] }
			],
			'in_dim':self.DataSets[self.data_names[0]][0],
			'out_dim':self.DataSets[self.data_names[0]][1],
		}

		# Initialize content dict
		self.content = copy.deepcopy(self.init_content)
		self.debug = False
		self.initUI()

	def initUI(self):
		# select widget layout
		self.toplayout = QGridLayout()

		# layout unit length
		self.h_fac = h_fac = 2
		self.w_fac = w_fac = 6

		# text font
		self.font = QFont("Ubuntu Mono", 10)
		self.font.setBold(True)

		# net type choose combobox
		self.net_choose_label = QLabel('NN Type:', self)
		self.net_choose_label.setFont(self.font)
		self.choose_net_comb = QComboBox(self)
		self.choose_net_comb.addItems(self.net_types)

		self.toplayout.addWidget(self.net_choose_label, 0,0, h_fac, w_fac-1)
		self.toplayout.addWidget(self.choose_net_comb, 0, w_fac-1, h_fac, w_fac)

		# dataset choose combobox
		self.dataset_label = QLabel('Data:', self)
		self.dataset_label.setFont(self.font)
		self.data_choose_comb = QComboBox(self)
		self.data_choose_comb.addItems(self.data_names)

		self.toplayout.addWidget(self.dataset_label, 0, 2*w_fac, h_fac, w_fac-2)
		self.toplayout.addWidget(self.data_choose_comb, 0, 3*w_fac-2, h_fac, w_fac)

		# optimizer choose combobox
		self.optim_label = QLabel('Optim:', self)
		self.optim_label.setFont(self.font)
		self.optim_comb = QComboBox(self)
		self.optim_comb.addItems(self.optim_names)

		self.toplayout.addWidget(self.optim_label, h_fac,0, h_fac, w_fac-1)
		self.toplayout.addWidget(self.optim_comb, h_fac, w_fac-1, h_fac, w_fac)

		# loss type choose combobox
		self.loss_label = QLabel('Loss:', self)
		self.loss_label.setFont(self.font)
		self.loss_comb = QComboBox(self)
		self.loss_comb.addItems(self.loss_types)

		self.toplayout.addWidget(self.loss_label, h_fac, 2*w_fac, h_fac, w_fac-2)
		self.toplayout.addWidget(self.loss_comb, h_fac, 3*w_fac-2, h_fac, w_fac)

		# batch size input line
		self.batch_label = QLabel('Batch size:', self)
		self.batch_label.setFont(self.font)
		self.batch_edit = QLineEdit(self)

		self.toplayout.addWidget(self.batch_label, h_fac*2, 0, h_fac, w_fac)
		self.toplayout.addWidget(self.batch_edit, h_fac*2, w_fac, h_fac, w_fac-1)

		# decay rate input line
		self.wdecay_label = QLabel('Weight decay:', self)
		self.wdecay_label.setFont(self.font)
		self.wdecay_edit = QLineEdit(self)

		self.toplayout.addWidget(self.wdecay_label, h_fac*2, w_fac*2, h_fac, w_fac)
		self.toplayout.addWidget(self.wdecay_edit,  h_fac*2, w_fac*3, h_fac, w_fac-2)

		# learning rate input line
		self.lr_label = QLabel('lr:', self)
		self.lr_label.setFont(self.font)
		self.lr_edit = QLineEdit(self)

		self.toplayout.addWidget(self.lr_label, h_fac*3, 0, h_fac, w_fac//3*2-2)
		self.toplayout.addWidget(self.lr_edit, h_fac*3, w_fac//3*2-2, h_fac, w_fac//3*2)

		# beta1 input line
		self.beta1_label = QLabel('Beta1:', self)
		self.beta1_label.setFont(self.font)
		self.beta1_edit = QLineEdit(self)

		self.toplayout.addWidget(self.beta1_label, h_fac*3, w_fac//3*4, h_fac, w_fac//3*2-1)
		self.toplayout.addWidget(self.beta1_edit, h_fac*3, w_fac//3*6-1, h_fac, w_fac//3*2)

		# beta2 input line
		self.beta2_label = QLabel('Beta2:', self)
		self.beta2_label.setFont(self.font)
		self.beta2_edit = QLineEdit(self)

		self.toplayout.addWidget(self.beta2_label, h_fac*3, w_fac//3*2*4, h_fac, w_fac//3*2-2)
		self.toplayout.addWidget(self.beta2_edit, h_fac*3, w_fac//3*2*5-2	, h_fac, w_fac//3*2)

		# add layer button
		self.add_layer_but = QPushButton( qtawesome.icon('fa.plus-circle',color='green'),'',parent = self)
		# substract layer button
		self.sub_layer_but = QPushButton( qtawesome.icon('fa.minus-circle',color='red'), '',parent = self)

		# connect add, sub button signal
		self.add_layer_but.clicked.connect(self.add_hidden_layer)
		self.sub_layer_but.clicked.connect(self.del_hidden_layer)

		self.toplayout.addWidget(self.add_layer_but, h_fac*4, 0, h_fac, w_fac//2)
		self.toplayout.addWidget(self.sub_layer_but, h_fac*4, w_fac//2, h_fac, w_fac//2)

		# Design first layer bar
		# input layer labels
		self.layer_labels = [
			QLabel('Layer', parent = self),
			QLabel('Input', parent = self),
			QLabel('Output', parent = self),
			QLabel('Bias', parent = self),
			QLabel('Activation', parent = self),
			QLabel('Dropout', parent = self),
		]
		for i,label in enumerate(self.layer_labels):
			label.setFont(self.font)
			label.setFrameShape(QFrame.Panel)
			self.toplayout.addWidget(label, h_fac*5, w_fac//3*2*i, h_fac//2, w_fac//3*2)

		# input layer widgets
		self.input_layer_widgets = [
			QLabel('Input', parent = self),
			QLabel('%d'%self.content['layers'][0]['in'] ,parent=self),
			QLineEdit(self),
			QCheckBox(self),
			QComboBox(self),
			QLineEdit(self),
		]
		self.input_layer_widgets[4].addItems(self.Activations)
		for i,wid in enumerate(self.input_layer_widgets):
			self.toplayout.addWidget(wid, h_fac*6, w_fac//3*2*i, h_fac//2, w_fac//3*2)

		# layer widget dict save all layer related widgets
		self.layer_widget_dict = {0:self.input_layer_widgets}
		# the number of layers
		self.lnum = 1

		self.toplayout.setSpacing(0)
		self.setLayout(self.toplayout)
		self.setGeometry(300, 50, 100, 200)

	# Function connected to add layer button
	# add a hidden layer bar in UI
	def add_hidden_layer(self):
		layer_widgets = [
			QLabel('Layer%d'%self.lnum, parent = self),
			QLineEdit(self),
			QLineEdit(self),
			QCheckBox(self),
			QComboBox(self),
			QLineEdit(self),
		]
		layer_widgets[4].addItems(self.Activations)

		h_fac, w_fac = self.h_fac, self.w_fac
		for i,wid in enumerate(layer_widgets):
			self.toplayout.addWidget(wid, h_fac*(6+self.lnum), w_fac//3*2*i, h_fac//2, w_fac//3*2)

		# add it to over all layer related dict
		self.layer_widget_dict[self.lnum] = layer_widgets

		self.lnum += 1
		print('Now Layer dict keys',self.layer_widget_dict.keys())
		self.setLayout(self.toplayout)

	# delete a hidden layer bar in UI
	def del_hidden_layer(self):
		if self.lnum>1:
			print('Del %d'%(self.lnum-1))
			tmp_wid_list = self.layer_widget_dict[self.lnum-1]
			for wid in tmp_wid_list:
				# self.toplayout.removeWidget(wid)
				# sip.delete(wid)
				wid.deleteLater()
			self.layer_widget_dict.pop(self.lnum-1)
			print('After delete, keys are ',self.layer_widget_dict.keys())
			self.lnum -= 1
		else:
			print('Cannot delete Input layer')

	# Clear all hidden layer bar
	def clear_layers(self):
		if self.lnum > 1:
			for i in range(self.lnum-1):
				self.del_hidden_layer()
			self.content = copy.deepcopy(self.init_content)

	# The Function to constract the parameters dict
	def gene_content(self):
		# overall and optimization parameters
		self.content['net_type'] = self.choose_net_comb.currentText()
		self.content['dataset'] = self.data_choose_comb.currentText()
		self.content['optim_para']['type'] = self.optim_comb.currentText()
		self.content['optim_para']['batch_size'] = except_type(istr=self.batch_edit.text(), otype = 'int', th = 1000, default = 50)
		self.content['optim_para']['loss'] = self.loss_comb.currentText()
		self.content['in_dim'] = self.DataSets[self.content['dataset']][0]
		self.content['out_dim'] = self.DataSets[self.content['dataset']][1]
		op_para = {}
		op_para['lr'] = except_type(istr=self.lr_edit.text(), otype='float', th=1, default=1e-3)
		op_para['weight_decay'] = except_type(istr=self.wdecay_edit.text(), otype='float', th=1, default=0.)
		op_para['beta1'] = except_type(istr=self.beta1_edit.text(), otype='float', th=1, default=0.9)
		op_para['beta2'] = except_type(istr=self.beta2_edit.text(), otype='float', th=1, default=0.99)
		self.content['optim_para']['param'] = op_para
		self.content['optim_para']['param']['betas'] = (op_para['beta1'], op_para['beta2'])

		# parameters for all layers
		last_out = except_type(istr=self.layer_widget_dict[0][2].text(), otype = 'int', th = 1e2, default = None)
		for i in range(0,self.lnum):
			layer = {}
			layer['id'] = i
			if i == 0:
				layer['in'] = self.DataSets[self.content['dataset']][0]
			else:
				now_in = except_type(istr=self.layer_widget_dict[i][1].text(), otype = 'int', th = 1e2, default = None)
				# judge validity of input units number
				if now_in != last_out or not now_in:
					# if current layer unit number not equal to last layer out number
					# box warning and return none
					reply = QMessageBox.question(self, 'Warning',
					    "Number of input number in layer %d is incorrect!"%i, QMessageBox.Yes|
					    QMessageBox.No, QMessageBox.No )
					return None
				else:
					layer['in'] = now_in
			last_out = except_type(istr=self.layer_widget_dict[i][2].text(), otype = 'int', th = 1e2, default = None)
			if not last_out:
				reply = QMessageBox.question(self, 'Warning',
					    "Number of output number in layer %d is incorrect!"%i, QMessageBox.Yes|
					    QMessageBox.No, QMessageBox.No )
				return None
			else:
				layer['out'] = last_out

			# other parameters about this layer
			layer['activation'] = self.layer_widget_dict[i][4].currentText()
			layer['bias'] = True if self.layer_widget_dict[i][3].isChecked() else False
			layer['dropout'] = except_type(istr=self.layer_widget_dict[i][5].text(), otype = 'float', th = 1, default = 0.0)
			if len(self.content['layers']) == i:
				self.content['layers'].append(layer)
			else:
				self.content['layers'][i] = layer
		return self.content

	# clear all input
	def clear_all_input(self):
		self.content = copy.deepcopy(self.init_content)
		self.hidden_input.setText('')
		self.layer_num_input.setText('')
		self.dropout_check.setChecked(False)
		self.l2_check.setChecked(False)


if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = SelectWiget()
	ex.setGeometry(300, 50, 700, 500)
	ex.show()
	sys.exit(app.exec_())
