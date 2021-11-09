# -*- coding: utf-8 -*-
# @Time    : 2021/11/6 14:09
# @Author  : HiQiang
# @github  : https://github.com/HiQiang
# @website : http://HiQiang.club/
# @email   : lq_sjtu@sjtu.edu.cn
# @Site    : 
# @File    : layers.py
# @Software: PyCharm

import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()  # 深拷贝
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        # self.original_x_shape = None
        # 关于权重和偏置的倒数
        self.dW = None
        self.db = None

    def forward(self, x):
        # self.original_x_shape = x.shape
        # x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.W, self.x) + self.b
        return out

    def backward(self, dout):  # dout 关于这一层的输出(out)的导数,由后一层反传而来
        dx = np.dot(self.W.T, dout)
        self.dW = np.dot(dout, self.x.T)
        self.db = np.sum(dout, axis=1)
        self.db = self.db.reshape([-1, 1])

        return dx


def softmax(x):
    x = x - np.max(x, axis=0)
    y = np.exp(x)/np.sum(np.exp(x), axis=0)
    return y


def cross_entropy(y, t):
    batch_size = y.shape[1]  # 列数
    delta = 1e-7
    return -np.sum(t*np.log(y))/batch_size


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[1]
        if self.t.size == self.y.size:  # 监督数据是 one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        # else:
        #     dx =
        return dx





















