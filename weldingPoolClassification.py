# -*- coding: utf-8 -*-
# @Time    : 2021/8/27 7:34
# @Author  : HiQiang
# @github  : https://github.com/HiQiang
# @website : http://HiQiang.club/
# @email   : lq_sjtu@sjtu.edu.cn
# @Site    : 
# @File    : Resnet_model.py
# @Software: PyCharm

import os  # Miscellaneous operating system interfaces 各种操作系统接口，一个API https://docs.python.org/3/library/os.html
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms  # common image transformations. most transformations accept PIL images
                                             # and tensor images https://pytorch.org/vision/stable/transforms.html
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # PIL Python Image Library
from torchvision import models
import torch.nn as nn
import torch

label_file = '../label/label.csv'  # 标签文件
img_Dir = '../images/'  # 图片文件夹， 注意最后加上的 '/'


class MyDataset(Dataset):  # 继承一个Dataset类
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_lables = pd.read_csv(annotations_file)  # 图片标签的注释文件， 输出为dataframe格式
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_lables.iloc[idx, 0])  # iloc[] 索引，
        image = Image.open(img_path, )
        label = self.img_lables.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.img_lables)


transform = transforms.Compose(
   [transforms.Resize([224, 224]), transforms.ToTensor()]   # Resize 注意[], ToTensor() 自动归一化
)

training_set = MyDataset(annotations_file=label_file, img_dir=img_Dir, transform=transform, target_transform=None)

training_dataloader = DataLoader(training_set, batch_size=24, shuffle=True)

for i, data in enumerate(training_dataloader, 0):
    for j in range(data[0].shape[0]):
        img_plot = plt.imshow(data[0][j, 0, :, :], cmap='gray')
        if data[1][j] == 0:  # python 无 switch case 语句
            plt.title("Class 1: Normal Penetration")
        elif data[1][j] == 1:
            plt.title("Class 2: Lack of Fusion")
        elif data[1][j] == 2:
            plt.title("Class 3: Sag Depression")
        plt.show()
        break
    break


model = models.resnet50(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(2048, 3)
print(model)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for batch, (X, y) in enumerate(training_dataloader):
    print('第%i个batch' % (batch+1))
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    loss_sum = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()*len(X)

        if batch % 4 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"loss_sum: {loss_sum:>7f}")
    return loss_sum/size


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += len(X) * loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


epochs = 200

training_Loss = np.zeros(epochs)
test_Loss = np.zeros(epochs)
Acc_Training = np.zeros(epochs)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    training_Loss[t] = train(training_dataloader, model, loss_fn, optimizer)
    test_Loss[t], Acc_Training[t] = test(training_dataloader, model, loss_fn)

print("Done!")


plt.plot(training_Loss)
plt.plot(test_Loss)
plt.plot(Acc_Training)
plt.legend(("Training loss", "Test loss", "Accuracy"))
plt.title("Accuracy vs Loss")
plt.xticks(range(0, epochs, int(epochs/10)))
plt.show()

