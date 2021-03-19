## 根据第1cnn复写的layer版本
1st cnn中，我使用了节点来构筑层，再用层构筑网络。这样有一个缺点就是手工起楼太麻烦了。第一层3个节点，第二层1个节点，我就必须手动写4个节点，然后再把每个节点的forwards和backwards的函数都写一遍，太麻烦了。这次的cnn在上一个版本做了改进，直接初始化一层，forward和backword也是整层的操作。这样就省去一个一个初始化节点的麻烦了，可以一次生成任意个节点。<br>
```
import numpy as np
import sklearn
from sklearn import datasets
import random     #用来打乱数据
from matplotlib import pyplot as plt # 用来画图，查看误差变化

sample = sklearn.datasets.load_iris()
datas = sample.data[:100] #由于我还不会/不知道 训练3个目标，先从2个目标练起
targets = sample.target[:100] #o4的值取0和1，可以表示2个物种。所以就用前100个数据
a = [x  for x in range(100)]
random.shuffle(a)
train = a[:80]
test = a[80:]
train_x = np.array([[np.append(m,1)] for m in datas[train]])
train_y = targets[train]
test_x = datas[test]
test_y = targets[test]

def sigmoid(x):
  return (1/(1+np.exp(-x)))

def deriv_sigmoid(x):
  fx = sigmoid(x)
  return (fx * (1-fx))

class layer:
  def __init__(self, row, col):
    self.weight = np.random.rand(row, col)
  def feedforwards(self, x):
    return (x.dot(self.weight))
  def adjust(self, w):
    self.weight = np.subtract(self.weight, w)

def mse_loss(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()

class myCNN:
  def __init__(self):
    self.layer1 = layer(5,3)
    self.layer2 = layer(4,1)
  def feedforwards(self, x):
    self.h1 = self.layer1.feedforwards(x)
    self.o1 = sigmoid(self.h1)
    self.o1 = np.append(self.o1,1)
    self.h2 = self.layer2.feedforwards(self.o1)
    self.o2 = sigmoid(self.h2)
    return (self.o2)
  def feedbackwards(self, x, y_true, learn_rate = 1):
    dMSE_do2 = -2*(y_true - self.o2) 
    l2 = np.array([dMSE_do2 * deriv_sigmoid(self.h2) * self.o1])
    w2 = self.layer2.weight[: -1]
    l1 = x.T * dMSE_do2 * deriv_sigmoid(self.h2) * w2.T * deriv_sigmoid(self.h1)
    self.layer1.adjust(l1 * learn_rate)
    self.layer2.adjust(l2.T  * learn_rate)
  def train(self, all_x_trues, all_y_trues):
    errorlist = []
    learn_rate = 0.5
    epochs = 100
    for epoch in range(epochs):
      for x_true, y_true in zip(all_x_trues, all_y_trues):
        self.feedforwards(x_true)
        self.feedbackwards(x_true, y_true, learn_rate)
      if epoch % 10 == 0:
        y_preds  = np.reshape([self.feedforwards(m) for m in all_x_trues], (1,-1))
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" %(epoch, loss))
        errorlist.append(loss)
    return (errorlist)

testcnn1 = myCNN()
el = testcnn1.train(train_x,train_y)
plt.plot(el)
plt.show()

```

第一个cnn写的是有毛病的，我找了一个差不多的代码放在最下方，没实验过。<br>
```
  1 import numpy as np
  2 
  3 def sigmoid(x):
  4   # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  5   return 1 / (1 + np.exp(-x))
  6 
  7 def deriv_sigmoid(x):
  8   # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  9   fx = sigmoid(x)
 10   return fx * (1 - fx)
 11 
 12 def mse_loss(y_true, y_pred):
 13   # y_true and y_pred are numpy arrays of the same length.
 14   return ((y_true - y_pred) ** 2).mean()
 15 
 16 class OurNeuralNetwork:
 17   '''
 18   A neural network with:
 19     - 2 inputs
 20     - a hidden layer with 2 neurons (h1, h2)
 21     - an output layer with 1 neuron (o1)
 22 
 23   *** DISCLAIMER ***:
 24   The code below is intended to be simple and educational, NOT optimal.
 25   Real neural net code looks nothing like this. DO NOT use this code.
 26   Instead, read/run it to understand how this specific network works.
 27   '''
 28   def __init__(self):
 29     # Weights
 30     self.w1 = np.random.normal()
 31     self.w2 = np.random.normal()
 32     self.w3 = np.random.normal()
 33     self.w4 = np.random.normal()
 34     self.w5 = np.random.normal()
 35     self.w6 = np.random.normal()
 36 
 37     # Biases
 38     self.b1 = np.random.normal()
 39     self.b2 = np.random.normal()
 40     self.b3 = np.random.normal()
 41 
 42   def feedforward(self, x):
 43     # x is a numpy array with 2 elements.
 44     h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
 45     h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
 46     o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
 47     return o1
 48 
 49   def train(self, data, all_y_trues):
 50     '''
 51     - data is a (n x 2) numpy array, n = # of samples in the dataset.
 52     - all_y_trues is a numpy array with n elements.
 53       Elements in all_y_trues correspond to those in data.
 54     '''
 55     learn_rate = 0.1
 56     epochs = 1000 # number of times to loop through the entire dataset
 57 
 58     for epoch in range(epochs):
 59       for x, y_true in zip(data, all_y_trues):
 60         # --- Do a feedforward (we'll need these values later)
 61         sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
 62         h1 = sigmoid(sum_h1)
 63 
 64         sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
 65         h2 = sigmoid(sum_h2)
 66 
 67         sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
 68         o1 = sigmoid(sum_o1)
 69         y_pred = o1
 70 
 71         # --- Calculate partial derivatives.
 72         # --- Naming: d_L_d_w1 represents "partial L / partial w1"
 73         d_L_d_ypred = -2 * (y_true - y_pred)
 74 
 75         # Neuron o1
 76         d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
 77         d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
 78         d_ypred_d_b3 = deriv_sigmoid(sum_o1)
 79 
 80         d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
 81         d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
 82 
 83         # Neuron h1
 84         d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
 85         d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
 86         d_h1_d_b1 = deriv_sigmoid(sum_h1)
 87 
 88         # Neuron h2
 89         d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
 90         d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
 91         d_h2_d_b2 = deriv_sigmoid(sum_h2)
 92 
 93         # --- Update weights and biases
 94         # Neuron h1
 95         self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
 96         self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
 97         self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
 98 
 99         # Neuron h2
100         self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
101         self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
102         self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
103 
104         # Neuron o1
105         self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
106         self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
107         self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
108 
109       # --- Calculate total loss at the end of each epoch
110       if epoch % 10 == 0:
111         y_preds = np.apply_along_axis(self.feedforward, 1, data)
112         loss = mse_loss(all_y_trues, y_preds)
113         print("Epoch %d loss: %.8f" % (epoch, loss))
114 
115 # Define dataset
116 data = np.array([
117   [-2, -1],  # Alice
118   [25, 6],   # Bob
119   [17, 4],   # Charlie
120   [-15, -6], # Diana
121 ])
122 all_y_trues = np.array([
123   1, # Alice
124   0, # Bob
125   0, # Charlie
126   1, # Diana
127 ])
128 
129 # Train our neural network!
130 network = OurNeuralNetwork()
131 network.train(data, all_y_trues)
```
