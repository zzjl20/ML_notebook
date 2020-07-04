## [Google Colab简要教程](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d)
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
这样， 有命令行，有ipynb环境，就可以像本地一样使用了。<br>
比较重要的功能还有，就不一条一条翻译了：<br>
安装包<br>
设置GPU<br>
从github直接copy repo<br>
直接运行ipynb文件<br>
查看当前CPU/GPU/RAM<br>
等等，就不逐一翻译了。

