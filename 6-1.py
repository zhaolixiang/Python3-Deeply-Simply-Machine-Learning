# 导人数据集生成器
from sklearn.datasets import make_blobs
import matplotlib.pylab as plt

# 生成样本数为500 ，分类数为5 的数据集
data2 = make_blobs(n_samples=500, centers=5, random_state=8)
X2, y2 = data2

# 用散点图将数据集进行可视化
plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap=plt.cm.spring, edgecolor='k')
plt.show()


