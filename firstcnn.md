我写的第一个神经网络，不过并没有完成。我有个问题没法解决：误差总是固定在0.25左右，从不变化。我不知道该怎么办。但是暂时先把已经写的东西记下来吧。<br>
我好像明白是怎么回事了。我每次都用一个数据训练模型，那么训练好的模型对当前训练的数据误差为0，对下一类数据，训练又是0.25。应该一次训练，数据起码包含2种数据，这样训练出来的模型对两种数据综合误差最小。<br>
### 建立模型： 
<img src = "./pics/autodraw 6_28_2020.png"><br>
其中每个节点： <br>
x<sub>i</sub>w<sub>i</sub>+b<sub>i</sub> = h<sub>i</sub><br>
<img src="./pics/CodeCogsEqn (1).gif"><br>
具体写出来：<br>
<img src="./pics/Screen Shot 2020-06-29 at 12.22.29.png"><br>
损失函数：<br>
<img src="./pics/Screen Shot 2020-06-29 at 0.01.30.png"><br>
### 数据训练的流程
将一个4维向量 *x<sub>i</sub>* 投入到上述网络中，就会得到一个输出 *o<sub>4</sub>*,这便是此神经网络对这组数据的预测。当然预测不一定准确，要比对实际值*y*, 即看损失函数。然后通过调整参数，即w和b改善使得综合误差最小.损失函数还能告知w和b的改善方向，只要损失函数对w或者b求微分，得到的向量损失增长最快的方向，那么相反的方向就是下降最快的方向。
<img src="./pics/Screen Shot 2020-06-28 at 23.41.46.png"><br>
假设对w<sub>41</sub>求偏导数，有：<br>
<img src="./pics/Screen Shot 2020-06-29 at 0.12.47.png"><br>
将3个偏微分代入，可得对w<sub>41</sub>的偏导数。同理w<sub>42</sub>,w<sub>43</sub>,b<sub>41</sub>都可求：
<img src="./pics/Screen Shot 2020-06-29 at 0.27.04.png"><br>
由上式可知，出了最终结果o<sub>4</sub>, 中间变量h<sub>4</sub>,o<sub>1,2,3</sub>,这些也也会用到，所以之前的运算要记录这些变量。<br>
对第一列节点w<sub>1</sub>,b<sub>1</sub>的偏导数同理，继续求偏导数即可。<br>
<img src="./pics/Screen Shot 2020-06-29 at 9.49.45.png"><br>
<img src="./pics/Screen Shot 2020-06-29 at 10.38.48.png"><br>
<img src="./pics/Screen Shot 2020-06-29 at 10.39.26.png"><br>
同理n2,n3节点也可求4个w1个b，不再多说。以上等式可以看到，所有的h和o都是中间变量，在反向回推的时候需要用到，要把这些变量记录下来。如下图，红色部分的变量需要记录。<br>
<img src="./pics/autodraw 6_29_2020.png"><br>
### 代码：
导入包和数据：
```
import numpy as np
import sklearn
from sklearn import datasets
import random     #用来打乱数据
from matplotlib import pyplot as plt # 用来画图，查看误差变化

sample = sklearn.datasets.load_iris()
datas = sample.data[:100] #由于我还不会/不知道 训练3个目标，线从2个目标练起
targets = sample.target[:100] #o4的值取0和1，可以表示2个物种。所以就用前100个数据
a = [x  for x in range(100)]
random.shuffle(a)
train = a[:80]
test = a[80:]
train_x = datas[train]
train_y = targets[train]
test_x = datas[test]
test_y = targets[test]
```
定义f(x)及导数:
```
def sigmoid(x):
    return ( 1/(1+np.exp(-x)))
    
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return (fx * (1-fx))
```
定义节点:
```
class neuron:
    def __init__(self, n):  #n表示可以接受几个x做输入
        w = [np.random.normal() for _ in range(n)]  #随机初始化w
        b = np.random.normal()    #和b
        self.weight = w
        self.bias = b
     def feedforward(self, x):      # 一个节点最基本的功能，将4个x运算成一个输出。
        result = np.dot(x,self.weight) + self.bias
        #print("calculate: ", x ,self.weight, self.bias, result)
        return (sigmoid(result), result)          #要记录o和h，作为之后反向推要用
     def adjust(self, w,b):  #提供调整w和b的函数
        self.weight = w
        self.bias = b      
 ```
 有了节点，可以做网络了。
 ```
 class myCNN:
    def __init__(self):    #初始化4个节点,n1,n2,n3是第一层，n4用第一层的结果为输入
        self.n1 = neuron(4)
        self.n2 = neuron(4)
        self.n3 = neuron(4)
        self.n4 = neuron(3)
    def feedforward(self, x):
        o1,h1 = self.n1.feedforward(x) # 
        o2,h2 = self.n2.feedforward(x)
        o3,h3 = self.n3.feedforward(x)
        o4,h4 = self.n4.feedforward([o1,o2,o3])
        return (o1,o2,o3,o4,h1,h2,h3,h4)
 ```
 暂时写这么多。cnn的train函数好像有误，我去重写.<br>
