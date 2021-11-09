# -*- coding: utf-8 -*-
# @Time    : 2021/11/9 9:49
# @Author  : HiQiang
# @github  : https://github.com/HiQiang
# @website : http://HiQiang.club/
# @email   : lq_sjtu@sjtu.edu.cn
# @Site    : 
# @File    : weight_init_activation_histogram.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def Relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


x = np.random.rand(100, 1000)  # 100维的数据,1000个

node_num = 100  # 各隐藏层的神经元数量
hidden_layer_size = 5
activations = dict()  # 用于保存每层激活值

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.uniform(node_num, node_num) * 0.005

    # Xavier 初始值 sigmoid
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)

    # He 初始值
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
    z = np.dot(w, x)
    # a = sigmoid(z)
    a = Relu(z)
    activations[i] = a


# 绘制直方图
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0:
        plt.yticks([], [])
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()




