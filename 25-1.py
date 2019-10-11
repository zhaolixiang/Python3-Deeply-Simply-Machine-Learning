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
