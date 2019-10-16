# 导入MLP神经网络
from sklearn.neural_network import MLPClassifier
# 导入红酒数据集
from sklearn.datasets import load_wine
# 导人数据集拆分工具
from sklearn.model_selection import train_test_split

wine = load_wine()
X = wine.data[:, : 2]
y = wine.target
# 下面我们拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 接下来定义分类器
mlp = MLPClassifier(solver='lbfgs')
mlp.fit(X_train, y_train)

# 导入画图工具
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# 使用不同色块表示不同分类
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FFOOOO', '#OOFFOO', '#OOFFOO'])
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))

Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# 将数据特征用散点图表示出来
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=60)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
# 设定图题
plt.title("MLPClassifier : solver=lbfgs")
# 显示图形
plt.show()
