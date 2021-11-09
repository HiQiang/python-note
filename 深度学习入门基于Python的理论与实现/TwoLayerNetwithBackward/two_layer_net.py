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

from layers import *
from collections import OrderedDict


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(hidden_size, input_size)
        self.params['b1'] = np.zeros([hidden_size, 1])
        self.params['W2'] = weight_init_std * np.random.rand(output_size, hidden_size)
        self.params['b2'] = np.zeros([output_size, 1])

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        # return y

    def loss(self, x, t):
        y = self.predict(x)
        # return cross_entropy(y, t)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)

        y = np.argmax(y, axis=0)  # 按列作比较 返回最大值的索引 返回值为 np 数组类型 一维
        t = np.argmax(t, axis=0)

        accuracy = np.sum(y == t)/float(x.shape[1])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = dict()
        grads['W1'] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db
        return grads


