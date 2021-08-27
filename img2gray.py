import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import pandas as pd

img_path = '../images/0050.jpg'
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 将img转换为灰度图，返回一个nd array
img_gray_plot = plt.imshow(img, )
plt.show()


img_dir_path = '../images/'
files = os.listdir(img_dir_path)  # 遍历img_dir_path, 返回一个list
numbers_of_images = len(files)    # 返回files的长度，此处为文件夹下该文件的个数
image_data = np.zeros([numbers_of_images, 1, 800, 800])  # 为图片数据预分配空间
for i in range(0, numbers_of_images):  # 遍历每一张图片，转化为灰度图后，存入img_data
    img = cv2.imread(img_dir_path + files[i])
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image_data[i, 0, :, :] = img_gray

    # img = np.transpose(img, (2, 0, 1))
    # image_data[i, 0:3, :, :] = img


img_plot = plt.imshow(image_data[10, 0, :, :], cmap='gray')
plt.show()

label_Path = '../label/label.csv'


class MyDataset(Dataset):
    def __init__(self, img_data, label_path):
        label_df = pd.read_csv(label_path, header=None, sep='\t')  # 表示没有表头
        self.label = label_df[0]
        self.data = img_data
        self.length = label_df.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.length


data = MyDataset(img_data=image_data, label_path=label_Path)
training_loader = DataLoader(data, batch_size=10, shuffle=True)


for i, data in enumerate(training_loader, 0):
    for j in range(data[0].shape[0]):
        img_plot = plt.imshow(data[0][j, 0, :, :], cmap='gray')
        plt.show()
        print(data[1][j])
        print(j)
    print(i)
    break
