# 导人数据集生成器
from sklearn.datasets import make_blobs
# 导人KNN 分类器
from sklearn.neighbors import KNeighborsClassifier
# 导人画图工具
import matplotlib.pyplot as plt

# 导人数据集拆分工具
from sklearn.model_selection import train_test_split

# 生成样本数为200 ，分类为2 的数据集
data = make_blobs(n_samples=200, centers=2, random_state=8)
X, y = data
print('x', X, 'y', y)
# 将生成的数据集进行可视化
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spring, edgecolor='k')
plt.show()

import numpy as np
import matplotlib as mpl

clf = KNeighborsClassifier()
clf.fit(X, y)

# 下面的代码用于画图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, .2),
                     np.arange(y_min, y_max, .2))


Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = Z.reshape(xx.shape)


cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])
plt.pcolormesh(xx, yy, z, cmap=cm_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spring, edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classifier: KNN")

# 把新的数据点用五星表示出来
plt.scatter(6.75 , 4.82 , marker='*', c ='red', s=200)


# 对新数据点分类进行判断
print('代码运行结果')
print('＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝') # 打印分隔符让结果美观一些
print('新数据点的分类是:',clf.predict([[6.75,4.82]]))
print('＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝') # 打印分隔符让结果美观一些

plt.show()
