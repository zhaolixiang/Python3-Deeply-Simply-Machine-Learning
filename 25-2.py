# 导人numpy
import numpy as np
# 导入画图工具
import matplotlib.pyplot as plt
# 导入数据集生成工具
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=40, centers=2, random_state=50, cluster_std=2)
# 用散点图绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.cool)
# 显示图像
plt.show()

# 导入StandardScaler
from sklearn.preprocessing import StandardScaler

# 使用StandardScaler进行数据预处理
X_1 = StandardScaler().fit_transform(X)
# 用散点图绘制经过预处理的数据点
plt.scatter(X_1[:, 0], X_1[:, 1], c=y, cmap=plt.cm.cool)
# 显示图像
plt.show()

# 导入MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# 使用M工nMaxScaler进行数据预处理
X_2 = MinMaxScaler().fit_transform(X)
# 绘制散点图
plt.scatter(X_2[:, 0], X_2[:, 1], c=y, cmap=plt.cm.cool)
# 显示图像
plt.show()

# 导入RobustScaler
from sklearn.preprocessing import RobustScaler

# 使用RobustScale r进行数据预处理
X_3 = RobustScaler().fit_transform(X)
# 绘制散点图
plt.scatter(X_3[:, 0], X_3[:, 1], c=y, cmap=plt.cm.cool)
# 显示图像
plt.show()

# 导人Normalizer
from sklearn.preprocessing import Normalizer

# 使用Normalizer进行数据预处理
X_4 = Normalizer().fit_transform(X)
# 绘制散点图
plt.scatter(X_4[:, 0], X_4[:, 1], c=y, cmap=plt.cm.cool)
# 显示图像
plt.show()
