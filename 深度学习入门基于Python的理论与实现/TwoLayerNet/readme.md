两层神经网络

采用数值微分方法求解梯度

不是反向传播

CPU计算，计算量大，对每一个权重或者偏置计算数值微分都需要算两次前向传播

MNIST数据集

训练和测试只取了原数据集的一部分  training_sizr = 1000  test_size = 1000


batch_size = 500

xunl

最后准确率 training_accuracy = 0.964  test_accuracy = 0.853

