## [Google Colab简要教程](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d) [(中文翻译)](https://juejin.im/post/5c05e1bc518825689f1b4948)
中文翻译
Google Colab是谷歌云硬盘上的一个服务，他可以让云硬盘像本地一样，运行类似jupyter lab一样的界面，运行ipynb程序。更重要的是，Google Colab可以免费使用云GPU或者TPU，还预装了tensorflow最新版。下面就记录几个重要的命令<br>
#### 如何开始一个ipynb的项目？ ####
"New" -> "More" -> "Connect more Apps" -> search Google Colab. 然后"New" -> "More" -> “Google Colabatory”就可以开始了。<br>
#### 如何像本地那样，从terminal安装包？ ####
```
from google.colab import drive
drive.mount('/content/drive/')
```
这样就可以直接看到google云硬盘里的内容。导入了drive包之后，就可以直接在命令框里运行pip，ls等命令。然后安装keras等就简单了。如果想要更多的terminal权限，可以导入
```
import os
os.chdir("drive/app")
```
这样， 有命令行，有ipynb环境，就可以像本地一样使用了。**注意命令行之前要加"!",以免被系统认成python3 命令，就会出错。**"!"就会告诉系统这是个terminal命令。当然有时候不加也能识别出来，但是大堆的场合，识别不出来。<br>
查看GPU的加速效果：
```
import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
  
# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
gpu()

# Run the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))
```
结果
```
Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images (batch x height x width x channel). Sum of ten runs.
CPU (s):
3.862475891000031
GPU (s):
0.10837535100017703
GPU speedup over CPU: 35x
```
比较重要的功能还有，就不一条一条翻译了：<br>
安装包（!pip）<br>
设置GPU<br>
```
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```
从github直接copy repo<br>
直接运行ipynb文件<br>
查看当前CPU/GPU/RAM(!nvidia-smi)<br>
等等，就不逐一翻译了。

