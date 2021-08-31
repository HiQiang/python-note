# _______________________________________________
# from __future__ import ****

from __future__ import print_function
# 在python2.X中，像python3.X那样使用print()语法

# python2.7
print "Hello world"

# python3
print("Hello world")

from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement # 等等


# _______________________________________________
# super()在单继承中的应用

# Python中类的初始化都是__init__(), 父类和子类的初始化方式都是__init__(), 
# 如果子类初始化时没有这个函数，那么他将直接调用父类的__init__(); 
# 如果子类指定了__init__(), 就会覆盖父类的初始化函数__init__()，
# 如果想在进行子类的初始化的同时也继承父类的__init__(), 
# 就需要在子类中显示地通过super()来调用父类的__init__()函数

class Animal:  # 定义一个父类
    def __init__(self):  # 父类的初始化
        self.name = 'animal'
        self.role = 'parent'
        print('I am father')
 
class Dog(Animal):  # 定义一个继承Animal的子类
    def __init__(self):  # 子类的初始化函数，此时会覆盖父类Animal类的初始化函数
        super(Dog, self).__init__()  # 在子类进行初始化时，也想继承父类的__init__()就通过super()实现,此时会对self.name= 'animal'
        

        self.name = 'dog'  # 定义子类的name属性,并且会把刚才的self.name= 'animal'更新为'dog'
        print('I am son')
        
HiQiang = Dog() # 变量 HiQiang 为一个 Dog() 类 执行初始化函数 打印  I am father(由父类继承而来) I am son
print(HiQiang.name)#'dog'
print(HiQiang.role)#'parent'


# _______________________________________________
# if __name__ == '__main__':

# 一个python文件通常有两种使用方法，
# 第一是作为脚本直接执行，
# 第二是 import 到其他的 python 脚本中被调用（模块重用）执行
# 因此 if __name__ == '__main__': 的作用就是控制这两种情况执行代码的过程，
# 在 if __name__ == 'main': 下的代码只有在第一种情况下，作为脚本直接执行时才会被执行，
# 而 import 到其他脚本中是不会被执行的

# test.py
print('this is test')
if __name__ == '__main__':
    print('this is test2')
# out:
# this is test
# this is test2

# import_test.py
import test
# out:
# this is test

# if __name__=="__main__": 之前的语句被执行，之后的没有被执行
# 每个python模块(python文件，也就是此处的 test.py 和 import_test.py)都包含内置的变量 __name__
# 当该模块被直接执行的时候，__name__ 等于文件名(包含后缀 .py)；如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称(不包含后缀.py)
# 而 “__main__” 始终指当前执行模块的名称(包含后缀.py)
# 当模块被直接执行时，__name__ == '__main__' 结果为真
print(__name__)
# out:
# __main__


# _______________________________________________
# enumerate: 枚举
for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(dataloader):  # 每一步 loader 释放一小批(batch_size)数据用来学习
# step 对应第几批，
# (batch_x, batch_y)对应输入和target


# _______________________________________________
# python的print()字符串前面加f表示格式化字符串，
# 加f后可以在字符串里面使用用花括号括起来的变量和表达式，
# 如果字符串里面没有表达式，那么前面加不加f输出应该一样.
t = 1
print(f"Epoch {t + 1}\n-------------------------------")

# out
# Epoch 2
# -------------------------------



# _______________________________________________
# RGB图像转换为Gray图像
from PIl import Image
lenna = Image.open('lenna.png')  # 读取彩色图像
im = lenna.convert('L')          # 转变模式选择'L'

im_array = np.array(lenna)       # 图片格式变为ndarray变量
lenna = Image.fromarray(im_array)# ndarray变量变为图片格式，可用于PIL的进一步处理，比如灰度变换


# _______________________________________________
# pandas.DataFrame
# https://github.com/MorvanZhou/tutorials/tree/master/numpy%26pandas
import pandas as pd

s = pd.Series([1,3,6,np.nan,4,1]) # similar with 1D numpy
print(s)
# out
# 0    1.0
# 1    3.0
# 2    6.0
# 3    NaN
# 4    4.0
# 5    1.0
# dtype: float64
dates = pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['A','B','C','D'])
print(df['B'])
df2 = pd.DataFrame({'A' : 1.,
                       'B' : pd.Timestamp('20130102'),
                        'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                        'D' : np.array([3] * 4,dtype='int32'),
                        'E' : pd.Categorical(["test","train","test","train"]),
                        'F' : 'foo'})
print(df2)
print(df2.dtypes)
print(df.index)
print(df.columns)
print(df.values)
print(df.describe())
print(df.T)
print(df.sort_index(axis=1, ascending=False))
print(df.sort_values(by='B'))


# _______________________________________________
plt.legend() # 显示曲线label

list(range(1,20))
Out[33]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
plt.xticks(range(1,20)) # x坐标设置为整数


#________________________________________________
numpy numerical python

data = np.load("filename") # 加载文件
cmap # colour map

#————————————————————————————————————————————————
img_path = '../images'
files = os.listdir(img_path)
numbers_of_images = len(files)
image_data = np.zeros([numbers_of_images, 800, 800])



image = cv2.imread(img_path, )  # 输入图像是 HWC
image = image.transpose(2, 0, 1)  # 转换为  CHW


#————————————————————————————————————————————————
from torchvision import models              # 导入内置模型
model = torchvision.models.resnet50()      
print(model)


#———————————————————————————————————————————————
import torch.nn as nn           # 微调内置模型
model = models.resnet50(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(2048, 3)


#———————————————————————————————————————————————
import pandas as pd
a = pd.read_csv(tr_path)
