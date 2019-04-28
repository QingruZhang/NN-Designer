# -*- coding: utf-8 -*-
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


'''-----------------------------------------------------'''
if __name__ == '__main__':
    eg1 = {'id':0, 'in':5, 'out':8, 'bias':True, 'activation':'relu', 'dropout':0}
    eg2 = {'id':1, 'in':5, 'out':8, 'bias':True, 'activation':'relu', 'dropout':0}
    eg3 = {'id':2, 'in':5, 'out':8, 'bias':True, 'activation':'relu', 'dropout':0}
    print (get_code([eg1,eg2,eg3]))
    optimizer = {'type':SGD, 'param':{}, 'batch_size':5, 'loss':'MLE'}
    dataset = 'mnist'
    
    