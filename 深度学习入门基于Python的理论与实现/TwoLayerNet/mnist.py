# -*- coding: utf-8 -*-
# @Time    : 2021/11/4 9:47
# @Author  : HiQiang
# @github  : https://github.com/HiQiang
# @website : http://HiQiang.club/
# @email   : lq_sjtu@sjtu.edu.cn
# @Site    : 
# @File    : mnist.py
# @Software: PyCharm

import os
import urllib.request
import gzip
import numpy as np
import pickle

url_base = "http://yann.lecun.com/exdb/mnist/"
key_file = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz"
}
dataset_dir = os.path.dirname(os.path.abspath(__file__)) + "\\dataset"  # 获取当前文件所在文件夹路径
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "\\" + file_name
    if os.path.exists(file_path):
        return
    print("Downloading " + file_name + "...")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")


def download_mnist():
    for v in key_file.values():
        _download(v)


def _load_label(file_name):
    file_path = dataset_dir + "\\" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, mode="rb") as f:  # 不知道需不需要关闭
        labels = np.frombuffer(buffer=f.read(), dtype=np.uint8, offset=8)  # 返回 NumPy Array   offset 读取的起始位置，默认为0
        print("Done")
    return labels


def _load_img(file_name):
    file_path = dataset_dir + "\\" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, mode="rb") as f:
        data = np.frombuffer(buffer=f.read(), dtype=np.uint8, offset=16)  # offset 读取的起始位置，默认为0
    # data = data.reshape(-1, img_size)  # 此处可能需要更改
    data = data.reshape(img_size, -1, order="F")  # 784 行  order="F" 按列进行
    print("Done")

    return data


def _convert_numpy():
    dataset = dict()
    dataset["train_img"] = _load_img(key_file["train_img"])
    dataset["train_label"] = _load_label(key_file["train_label"])
    dataset["test_img"] = _load_img(key_file["test_img"])
    dataset["test_label"] = _load_label(key_file["test_label"])

    return dataset


def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, "wb") as f:
        pickle.dump(dataset, f, -1)  # pickle.dump(obj, file, protocol=None,)
    print("Done!")


def _change_one_hot_label(X):
    # T = np.zeros((X.size, 10))
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):  # 按行遍历 返回 (行)序号 和 值  行号是对应的标签序号
        row[X[idx]] = 1  # X[index] 返回标签   row[X[idx]] 是该行 标签值所对应的位置
    return T.T  # 转置 将标签转置为 列


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, mode="rb") as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    if one_hot_label:
        dataset["train_label"] = _change_one_hot_label(dataset["train_label"])
        dataset["test_label"] = _change_one_hot_label(dataset["test_label"])

    # 问题代码
    # if not flatten:
    #     for key in("train_img", "test_img"):
    #         dataset[key] = dataset[key].reshape(-1, 1, 28, 28, order="F")

    return(dataset["train_img"], dataset["train_label"]), (dataset["test_img"], dataset["test_label"])


if __name__ == '__main__':
    init_mnist()


