# -*- coding: utf-8 -*-
# @Time    : 2021/11/4 11:52
# @Author  : HiQiang
# @github  : https://github.com/HiQiang
# @website : http://HiQiang.club/
# @email   : lq_sjtu@sjtu.edu.cn
# @Site    : 
# @File    : train_neural_net.py
# @Software: PyCharm

from mnist import load_mnist
import numpy as np
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


train_loss_list = []
test_loss_list = []
train_accuracy_list = []
test_accuracy_list = []

iter_num = 10000
train_size = x_train.shape[1]
test_size = x_test.shape[1]
batch_size = 1000
learning_rate = 0.1

# batch_mask = np.random.choice(train_size, 20000)
# x_train = x_train.T[batch_mask].T  # 两次转置 权宜之计
# t_train = t_train.T[batch_mask].T

# batch_mask = np.random.choice(test_size, 1000)
# x_test = x_test.T[batch_mask].T  # 两次转置 权宜之计
# t_test = t_test.T[batch_mask].T

train_size = x_train.shape[1]

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
for i in range(iter_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train.T[batch_mask].T  # 两次转置 权宜之计
    t_batch = t_train.T[batch_mask].T

    # test_x_batch = x_test.T[batch_mask].T  # 两次转置 权宜之计
    # test_t_batch = t_test.T[batch_mask].T

    grad = network.gradient(x_batch, t_batch)
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    if (i + 1) % 2 == 0:
        train_loss = network.loss(x_batch, t_batch)
        train_loss_list.append(train_loss)
        test_loss = network.loss(x_test, t_test)
        test_loss_list.append(test_loss)
        train_accuracy = network.accuracy(x_train, t_train)
        train_accuracy_list.append(train_accuracy)
        test_accuracy = network.accuracy(x_test, t_test)
        test_accuracy_list.append(test_accuracy)
        print("TrainLoss: " + str(train_loss))
        print("TestLoss: " + str(test_loss))
        print("TrainAccuracy: " + str(train_accuracy))
        print("TestAccuracy: " + str(test_accuracy))
    print(i)


plt.plot(train_loss_list, label="TrainLoss")
plt.plot(test_loss_list, label="TestLoss")
plt.legend()
plt.show()
plt.plot(train_accuracy_list, label="TrainAccuracy")
plt.plot(test_accuracy_list, label="TestAccuracy")
plt.legend()
plt.show()
