其实第二个使用tensor flow的模型并没有完成。它其实是失败了。步子迈大了。<br>
今天遇到了适合使用神经网络的例子，决定2个隐藏层，每个隐藏层100，10个节点。这样就没法按照1stcnn那样一个一个的设置节点了。必须按照层来初始化.<br>
前期包的导入和数据准备，仍然像1stcnn一样。<br>
```
import numpy as np
import sklearn
from sklearn import datasets
import random     #用来打乱数据
from matplotlib import pyplot as plt # 用来画图，查看误差变化

sample = sklearn.datasets.load_iris()
datas = sample.data[:100] 
targets = sample.target[:100] 
a = [x  for x in range(100)]
random.shuffle(a)
train = a[:80]
test = a[80:]
train_x = datas[train]
train_y = targets[train]
test_x = datas[test]
test_y = targets[test]
```
定义中间层，这时开始有变化：
```
class layer:   # 隐藏层版本。 与节点完全不相干
  def __init__(self, innum, outnum):   #随机初始化曾，需要给定输入多少个特征，输出多少个特征。
    self.weights = np.random.rand(innum+1, outnum)  #在给定的输入层数之后+1，做b
    print(self.weights)
  def feedforwards(self, x):   #向前传播，要将输入的x后面加一个"1"， 用俩和b相乘。
    x = np.append(x, 1)
    return (np.dot(x, self.weights))
```
下面是引用：没写完，以后再接着写
```
layerx = layer(4,3)
x1 = [4,3,2,1]
y = layerx.feedforwards(x1)
print((y))
```
