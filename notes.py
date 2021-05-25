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

# Python中类的初始化都是__init__(), 父类和子类的初始化方式都是__init__(), 
# 如果子类初始化时没有这个函数，那么他将直接调用父类的__init__(); 
# 如果子类指定了__init__(), 就会覆盖父类的初始化函数__init__()，
# 如果想在进行子类的初始化的同时也继承父类的__init__(), 
# 就需要在子类中显示地通过super()来调用父类的__init__()函数。


# _______________________________________________
# super()在单继承中的应用

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
