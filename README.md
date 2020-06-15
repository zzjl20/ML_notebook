# ML_notebook
Resource :<br>
https://developers.google.com/machine-learning/crash-course
## [Tensorflow play ground. ](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.33934&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&discretize_hide=false)
## [CNN 解释器](https://poloclub.github.io/cnn-explainer/)
## [神经网络初始化](https://www.deeplearning.ai/ai-notes/initialization/) 及 [中文解释](https://zhuanlan.zhihu.com/p/67149162)

## 机器学习的流程：
构建问题。<br>
找一下已经上标的数据集<br>
为你的模型设计数据<br>
设计你的数据在模型上的input<br>
设计一个容易使用的接口<br>
设计一个有质量的output<br>
澄清问题，设定目标，要解决什么问题-> 想象一下最理想的输出 -> 设定一下判断矩阵：如何判定模型成功，哪个指标来判断-> 你的output是什么？ -> 使用output，你的output可以回答什么问题？ ->这个问题，如果不用ML，使用普通编程，应该如何解决？<br>
解构你的问题-> 从简单情况开始分析->确认数据来源->为你的建模设计数据解构->确保使用比较容易的input->有学习能力-> 考虑一下可能的偏差<br>
|问题|描述|举例|
|--|--|--
|分类|预测属于N类中的哪一类|猫，狗，马，还是熊|
|回归|预测数字|点击率|
|聚类|预测组分类|搜索最相关文档|
|规则学习|寻找关联|如果你买了汉堡，八成也要接着买可乐|
|整体输出|比较复杂的结果输出|自然语言输出， 图片人脸识别|
|排序|预估此结果在整体中的比重|搜索结果排序|

线性模型需要上千个数据训练，神经网络需要十万数量级数据。不要使用机器学习来找有效特征。把所有特征都扔给模型的结果是训练出来的模型超级贵，特别复杂，而且总在一些鸡毛蒜皮的小特征上fit well。设计model要直接解决问题，而不是间接，或者是问题的前置问题。<br>
ML的输出，必须跟着决策！<br>
## ML高难问题
|聚类|异常检测|启发式|
|--|--|--|
|<img width="150" height="150" src="https://developers.google.com/machine-learning/problem-framing/images/LabeledClusters.svg"/>|<img width="150" height="150" src="https://developers.google.com/machine-learning/problem-framing/images/Anomaly.png"/>|客人买了书，是不是有可能因为他前几天刚看了这本书的书评？|
## 机器学习概念术语
<b>样本， 特征，标签</b>。样本是要处理的数据，可以写作 <i>X</i>,每个样本都由特征来描述，每个特征可以看作x。每个样本<i>X<sub>i</sub></i>可以看作由j个特征张起的空间中的点<i>X<sub>i</sub></i>（x<sub>i1</sub>,x<sub>i2</sub>,...,x<sub>ij</sub>）。标签可以看成Y。一个样本对应了一个标签，可以看成点X由映射f变换成一个值Y，即f（<i>X<sub>i</sub></i>） = Y。机器学习就是给出空间中大量的点<i>X</i>和映射值Y，来拟合f函数。这个f函数，称之为<b>模型</b>。向模型展示样本，逐渐求出模型的过程，叫<b>训练</b>或<b>学习</b>。给出没有标签的样本<i>X</i>,带入f函数求得对应的Y的过程，称作<b>预测</b>。本质上和解析几何曲线拟合一样。<br>
使用有标签的数据学习，拟合出函数f，称之为有<b>监督的学习</b>。数据无标签，那么就要总结这些数据的模式，称为<b>无监督学习</b><br>
<b>误差</b>预测的数值与实际数值的差值。由于预测数字有时高于实际数字而有时低于，那么使用这个差值的平方，不管误差为证为负统一换算成正值，更容易看出差值。这个平方值也叫<b>方差</b>，或者叫<b>偏差</b>。我们调整模型，使得总体的方差最小而非某一个样本的方差最小。<br>
<b>梯度</b>即与模型参数相关的误差函数的导数。梯度，有两种含义，这在一般教材中没有明确说明，似乎大家都觉得这么简单的事情不必细说。<b>梯度函数</b>，要确定某个空间曲面f(x,y)的梯度函数梯度,是f(x,y)对所有自变量的偏导数，是一个2维向量，每个分量是一个偏导函数。<b>某一点的梯度</b>是将点(x,y)代入到这个偏导数每个分量得到的向量，这个梯度是一个有着确定值的向量。例如：<br>
f(x,y) = 4+(x-2)<sup>2</sup>+2y<sup>2</sup>,它是一个山谷一样的曲面：<br>
<img width="400"  src="./pics/gradient.svg"/><br>
在(2,0)点有函数最小值4。它的梯度函数是（2x-4,4y）,在（x,y）=(0,0)点的梯度值是（-4，0）。 在阅读资料时碰到梯度，要结合上下文分析说得是梯度函数还是某点的梯度值。<br>
<b>梯度下降法</b>，把误差函数看成上文的f(x,y)，其中x,y是模型的参数。这样将建立函数参数与误差之间的联系。我们的目标是寻找函数的最小值，即最小误差那一点，此时参数是什么值。梯度的方向，是函数上升最快的方向，那么函数的反方向，即使函数下降最快的方向。<br>
比如上例，假设我们不知道函数在（2，0）点可以使得误差变最小，4。我们随机选择一点（0，0），此时误差为f(0,0)=8,(0,0)点的梯度为（-4，0），下降最快的方向是（4，0）。下一次试验的向量应该是（0，0）+n*(4,0)，这里n是步长。如果我们取n=1,那么下个试验向量为（4，0）。代入f中，f(4,0)=8,(4,0)点的梯度为（4，0），下降最快的方向是（-4,0)。那么再下一次的试验向量应该是（4，0）+n(-4,0)。<br>
实际上我们可以看到刚才那一步步长过大使其跨过了极值点，两次梯度的模方向相反，模长相同，暗示我们这一步正好跨到了极值两边相等的两个误差值。如果我们不取n=1,取其一半，则可以到达极值点。刚才那一步，若令n=0.5，则下一个测试向量为（0，0）+0.5* （4，0） = (2,0)。将（2，0）代入f中，误差f(2,0)=4。此时梯度为（0,0），梯度模长=0，说明此点是极值点。即当这个模型的参数(x,y)=(2,0)时，此模型达到最佳模拟，误差为4，是这个模型能做到的最佳性能了。


可以指示我们可以快速如何修改参数，使得模型的误差变小。修改后的模型，再代入

