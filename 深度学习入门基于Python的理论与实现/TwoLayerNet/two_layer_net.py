# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 10:09
# @Author  : HiQiang
# @github  : https://github.com/HiQiang
# @website : http://HiQiang.club/
# @email   : lq_sjtu@sjtu.edu.cn
# @Site    : 
# @File    : two_layer_net.py
# @Software: PyCharm

"""两层神经网络,手动搭建"""

import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def softmax(x):
    x = x - np.max(x, axis=0)  # 按列取最大值 防止溢出对策
    y = np.exp(x)/np.sum(np.exp(x), axis=0)  # 按列求和
    # c = np.max(x)  # 防止溢出对策
    # return np.exp(x - c)/np.sum(np.exp(x - c))
    return y


def cross_entropy(y, t):
    # 标签需是 one-hot 形式
    # if y.shape[1] == 1:  # 单个样本
    batch_size = y.shape[1]
    delta = 1e-7
    return -np.sum(t*np.log(y))/batch_size  # 元素位置相对应的运算 然后求和
    # cross_entropy


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tem_val = x[idx]
        x[idx] = float(tem_val) + h
        fxh1 = f(x)  # f(x + h)

        x[idx] = tem_val - h
        fxh2 = f(x)  # f(x - h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tem_val  # 还原值
        it.iternext()
    return grad


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(hidden_size, input_size)
        self.params['b1'] = np.zeros([hidden_size, 1])
        self.params['W2'] = weight_init_std * np.random.rand(output_size, hidden_size)
        self.params['b2'] = np.zeros([output_size, 1])

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # 前向传播
        a1 = np.dot(W1, x) + b1  # 对于批训练,此处运用了broadcast功能 a1 为向量或矩阵
        z1 = sigmoid(a1)  # sigmoid
        a2 = np.dot(W2, a1) + b2
        y = softmax(a2)  # softmax

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)

        y = np.argmax(y, axis=0)  # 按列作比较 返回最大值的索引 返回值为 np 数组类型 一维
        t = np.argmax(t, axis=0)

        accuracy = np.sum(y == t)/float(x.shape[1])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = dict()
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

#


# net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
#
# x = np.random.rand(784, 100)
# y = net.predict(x)
# t = np.random.rand(10, 100)
# grads = net.numerical_gradient(x, t)

