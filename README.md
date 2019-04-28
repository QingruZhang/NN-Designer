# ![avatar](img/Logo.png =20x20) NN-Designer 
## Author
+ Qingru Zhang


## Requirements

* Python >= 3.4.0 
* PyQt5 5.9
* Pytorch 1.0.1

### Installation ###
    pip install PyQt5
    pip3 install torchvision


##Getting started ![avatar](img/geneAct.png =20x20)  

### Brief Introduction
It is a software course design about Neural network visualization designer. PyQt5 is used to design GUI.
Pytorch is used to generate code and train corresponding neural networks.

###Run Code Directly ![avatar](img/trainAct.png =20x20)
For Windows or Linux paltform, you can run `main.py` after installing dependency packages.
In the this directory, run command:
```
    python3 main.py
```

###About File

| Dir   |      Comment      |
|----------|:-------------:|
|img/                   |    程序中使用的图片  |
|train_code/model_v3  | pytorch代码生成与训练模块 |
|main.py | 程序入口 |
|TopDesign.py | UI顶层设计 |
|TrainThread.py | 神经网络训练线程类定义 |
|SelectWidget.py| 神经网络参数构造与传递部件 |
|DrawNeuralNetwork.py| 神经网络结构绘制函数 |
|README.md                 |    说明文档    |

