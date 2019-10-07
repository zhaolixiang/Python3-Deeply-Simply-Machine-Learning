# 导人数据集生成器
from sklearn.datasets import make_blobs
import matplotlib.pylab as plt

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib as mpl

# 生成样本数为500 ，分类数为5 的数据集
data2 = make_blobs(n_samples=500, centers=5, random_state=8)
X2, y2 = data2

clf = KNeighborsClassifier()
clf.fit(X2, y2)

# 下面的代码用于画图
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = Z.reshape(xx.shape)
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#448ced', '#6600FF', '#FF00FF'])
plt.pcolormesh(xx, yy, z, cmap=cm_light)
# 用散点图将数据集进行可视化
plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap=plt.cm.spring, edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classifier : KNN")
plt.show()

print(' 代码运行结果')
print('==============================')
print('模型正确率{:.2f}'.format(clf.score(X2, y2)))
print('==============================')
