# -*- coding: utf-8 -*-

eg1 = {'id':0, 'in':5, 'out':8, 'bias':True, 'activation':'relu', 'dropout':0}
eg2 = {'id':1, 'in':8, 'out':7, 'bias':True, 'activation':'relu', 'dropout':0}
eg3 = {'id':2, 'in':7, 'out':6, 'bias':True, 'activation':'relu', 'dropout':0}
signal = [eg1, eg2, eg3]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_activation(string):
    if string == 'sigmoid':
        return nn.Sigmoid()
    else:
        if string == 'tanh':
            return nn.Tanh()
        else:
            return nn.ReLU()

def get_single_code(example):
    res_init = 'self.layer'+str(example['id'])+' = nn.Linear('+str(example['in'])+', '+str(example['out'])+', bias='+str(example['bias'])+')'
    if example['activation'] == 'relu':
        res_activation = 'self.activation'+str(example['id'])+' = nn.ReLU()' 
    if example['activation'] == 'tanh':
        res_activation = 'self.activation'+str(example['id'])+' = nn.Tanh()'
    if example['activation'] == 'sigmoid':
        res_activation = 'self.activation'+str(example['id'])+' = nn.Sigmoid()'
    res_dropout = 'self.dropout'+str(example['id'])+' = nn.Dropout('+str(example['dropout'])+')'
    res_forward = 'data = self.layer'+str(example['id'])+'(data)'+'\n        data = self.activation'+str(example['id'])+'(data)'+'\n        data = self.dropout'+str(example['id'])+'(data)\n'        
    return res_init, res_activation, res_dropout, res_forward

def data_generating(func_type, size):
    size = int(sqrt(size))
    if func_type == 'Beale Function':
        x1, x2 = np.meshgrid(np.linspace(-4.5, 4.5, size), np.linspace(-4.5, 4.5, size))
        x1 = torch.FloatTensor(x1.reshape(1, size*size)).permute(1,0)
        x2 = torch.FloatTensor(x2.reshape(1, size*size)).permute(1,0)
        data2 = torch.cat((x1, x2), 1)
        y = (1.5-x1+x1*x2)*(1.5-x1+x1*x2) + (2.25-x1+x1*x2*x2) \
            *(2.25-x1+x1*x2*x2) +(2.625-x1+x1*x2*x2*x2)*(2.625-x1+x1*x2*x2*x2)
        data2 = torch.cat((data2, y), 1)
    if func_type == 'Mulitiply Square':
        x1, x2 = np.meshgrid(np.linspace(-4.5, 4.5, size), np.linspace(-4.5, 4.5, size))
        x1 = torch.FloatTensor(x1.reshape(1, size*size)).permute(1,0)
        x2 = torch.FloatTensor(x2.reshape(1, size*size)).permute(1,0)
        data2 = torch.cat((x1, x2), 1)
        y = x1*x1+x2*x2
        data2 = torch.cat((data2, y), 1)
    if func_type == 'Himmelblau function':
        x1, x2 = np.meshgrid(np.linspace(-5, 5, size), np.linspace(-5, 5, size))
        x1 = torch.FloatTensor(x1.reshape(1, size*size)).permute(1,0)
        x2 = torch.FloatTensor(x2.reshape(1, size*size)).permute(1,0)
        data2 = torch.cat((x1, x2), 1)
        y = (x1*x1+x2-11)*(x1*x1+x2-11) + (x1+x2*x2-7)*(x1+x2*x2-7)
        data2 = torch.cat((data2, y), 1)
    if func_type == 'Levi function N.13':
        x1, x2 = np.meshgrid(np.linspace(-10, 10, size), np.linspace(-10, 10, size))
        x1 = torch.FloatTensor(x1.reshape(1, size*size)).permute(1,0)
        x2 = torch.FloatTensor(x2.reshape(1, size*size)).permute(1,0)
        data2 = torch.cat((x1, x2), 1)
        y = torch.sin(3*pi*x1)*torch.sin(3*pi*x1) + (x1-1)*(x1-1)*(1+torch.sin \
                     (3*pi*x2)*torch.sin(3*pi*x2)) + (x2-1)*(x2-1)*(1+torch.sin(2*pi*x2)*torch.sin(2*pi*x2))
        data2 = torch.cat((data2, y), 1)
    return data2

def get_code(eg_list):
    init = ''
    forward = ''
    for example in eg_list:
        init_i, activation_i, dropout_i, forward_i = get_single_code(example)
        init += '        '+ init_i +'\n        '+ dropout_i + '\n        ' + activation_i +'\n\n'
        forward += '        '+forward_i + '\n'
    
    result = 'class Network(nn.Module):\n    '+'def __init__(self):\n        '+'super(Network, self).__init__()\n'+init
    result += '    '+'def forward(self, data):\n'+forward+'        return data'
    return result
    "double w1 = [%d , %d]\n doubel w2 =  "%( w , ...)
    with open('') as f:
        f.write(str)

class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_list):
        num_layers = len(hidden_list)
        hidden_layers = hidden_list
        hidden_layers[0]['in'] = input_dim
        hidden_layers[num_layers-1]['out'] = output_dim
        
        super(Network, self).__init__()
        
        self.layer_stack = nn.ModuleList([])
        
        for layer in hidden_list:
            self.layer_stack.append(nn.Linear(layer['in'], layer['out'], bias=layer['bias']))
            self.layer_stack.append(get_activation(layer['activation']))
            self.layer_stack.append(nn.Dropout(layer['dropout']))
    def forward(self, data):
        for layer in self.layer_stack:
            data = layer(data)
        return data
    
optimizer = {'type':'SGD', 'param':{'lr':0.0001}, 'batch_size':1000, 'loss':'MSE'}
dataset = {'mnist', 'sgd'}

class Optimizer(object):
    def __init__(self, optimizer, param):
        if optimizer["type"] == 'SGD':
            self.optim = torch.optim.SGD(param, lr=optimizer['param']['lr'])
        if optimizer["type"] == "Adam":
            self.optim = torch.optim.Adam(param, lr=optimizer['param']["lr"], 
                                              betas=optimizer['param']["betas"])
        if optimizer['loss'] == 'MSE':
            self.loss_func = nn.MSELoss()
        if optimizer['loss'] == 'MAE':
            self.loss_func = nn.L1Loss()
        self.batch_size = optimizer['batch_size']
            
    def zero_grad(self):
        self.optim.zero_grad()
    
    def update_params(self):
        self.optim.step()
        
    def calculate_loss(self, prediction, data):
        loss = self.loss_func(prediction, data)
        return loss
    
class Visualized_Trainer(object):
    def __init__(self, optimizer, input_dim, output_dim, hidden_list, dataset, 
                 loss_fig, out_fig, update_Carve_signal=None, update_num = 1):
        self.network = Network(input_dim, output_dim, hidden_list)
        self.optim = Optimizer(optimizer, self.network.parameters())
        self.batch_size = self.optim.batch_size
        self.parameters = list(self.network.parameters())
        self.checkpoint = 0
        self.data = dataset
        self.data_size = self.data.size()[0]
        self.size = int(sqrt(self.data_size))
        self.current_epoches = 0
        self.loss_history = np.array([])
        self.update = 1
        # self.out_ax = Axes3D(fig)
        self.update_num = update_num
        self._new_epoches = 1
        
        self.join = False
        self.loss_fig = loss_fig
        plt.figure(self.loss_fig.number)
        plt.clf()
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_ax.set_xlabel('Epoches')
        self.loss_ax.set_ylabel('Loss')
        # self.loss_ax.clear()

        # self.out_fig = plt.figure()
        self.out_fig = out_fig
        plt.figure(self.out_fig.number)
        plt.clf()
        self.out_ax = Axes3D(self.out_fig)
        # self.out_ax.clear()
        self.update_Carve_signal = update_Carve_signal
        
    def get_batch(self):
        batch_data = torch.FloatTensor([0])
        if self.checkpoint + self.batch_size < self.data_size:
            batch_data = self.data[self.checkpoint:self.checkpoint + self.batch_size]
            self.checkpoint += self.batch_size
            self.update = 0
            self._new_epoches = 0
        if self.checkpoint + self.batch_size >= self.data_size:
            batch_data = self.data[self.checkpoint:]
            self.checkpoint = 0
            self.current_epoches += 1
            self._new_epoches = 1
        return batch_data
        
    def visualized_training(self, epoches):
        self.current_epoches = 0
        self.loss_history = np.array([])
        self.plot_epoches = np.array([])
        plt.ion()
        while self.current_epoches < epoches and not self.join:
            
            if self._new_epoches == 1:
                dataxx = self.data[:,0:2]
                
                labels = self.data[:,2:3]
                predictionx = self.network(dataxx)
                loss_value = float(self.optim.loss_func(predictionx, labels))
                self.loss_history = np.append(self.loss_history, loss_value)
                self.plot_epoches = np.append(self.plot_epoches, self.current_epoches)
                self.update_loss_plot(epoches)
            if self.current_epoches % self.update_num == 1 + self.update_num:
                self.update_predict_plot(predictionx, dataxx)
                
            data = self.get_batch()
            datax = data[:,0:2]
            datay = data[:,2:3] 
            prediction = self.network(datax)
            loss = self.optim.loss_func(prediction, datay)
            if self.current_epoches%self.update_num in [0,1]:
                print('Epoch %d loss is %.2f'%(self.current_epoches,float(loss)))
            #Back Propogation
            self.optim.zero_grad()
            loss.backward()
            self.optim.optim.step()
                
    def update_loss_plot(self, epoches):
        self.loss_ax.plot(self.plot_epoches, self.loss_history, 'r-', lw=2)
        self.loss_ax.set_xlim(0, epoches+2)
        if __name__ == '__main__':
            plt.pause(0.1)
            print('plot loss pause at epoch %d'%self.current_epoches)
        else:
            print('plot loss at epoch %d'%self.current_epoches)
            self.update_Carve_signal.emit('loss')
        
    def update_predict_plot(self, prediction, datax):
        x1 = datax[:,0:1].detach().numpy()
        x2 = datax[:,1:2].detach().numpy()
        x1 = x1.reshape(self.size,self.size)
        x2 = x2.reshape(self.size,self.size)
        prediction = prediction.detach().numpy().reshape(self.size,self.size)
        self.out_ax.plot_surface(x1, x2, prediction, rstride=1, cstride=1, cmap='rainbow')
        if __name__ == '__main__1':
            plt.pause(0.1)
            print('plot out pause at %d'%self.current_epoches)
        else:
            print('plot out at %d'%self.current_epoches)
            self.update_Carve_signal.emit('out')
        
    def update_original_plot(self, labels, datax):
        plt.cla()
        x1 = datax[:,0:1]
        x2 = datax[:,1:2]
        ax=plt.subplot(111,projection='3d') 
        ax.scatter(x1.numpy(), x2.numpy(), labels.numpy(), c="r")
        plt.pause(1)

if __name__ == '__main__':
    fake_data = data_generating('Mulitiply Square', 2500)
    plt.close('all')
    loss_fig = plt.figure(1)
    out_fig = plt.figure(2)
    loss_fig.show()
    out_fig.show()
    trainer = Visualized_Trainer(optimizer, 2, 1, signal, fake_data, loss_fig, out_fig, )
    trainer.visualized_training(200)
    
    
    