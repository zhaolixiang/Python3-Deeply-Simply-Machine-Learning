# 导人多项式朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
# 导入数据预处理工具MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# 生成样本数量为500 ，分类数为5的数据集
X, y = make_blobs(n_samples=500, centers=5, random_state=8)
# 将数据集拆分成训练集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

# 使用MinMaxScaler对数掘进行预处理，使数据全部为非负值
scaler = MinMaxScaler()
scaler.fit(X_train)
scaler.fit(X_test)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 用多项式朴素贝叶斯拟合数据
mnb = MultinomialNB()
mnb.fit(X_train_scaled, y_train)

# 限定横轴与纵轴的最大值
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# 用不同的背景色表示不同的分类
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))

# 用不同色块来表示不同的分类
z = mnb . predict(np . c_[(xx . ravel() , yy . ravel())]) . reshape (xx.shape)
plt . pcolormesh(xx , yy, z , cmap=plt.cm . Pastel1)
# 用散点图画出训练集和测试集数据
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.cool, edgecolors='k')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.cool, marker='*', edgecolors='k')
# 设定横纵轴范围
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# 定义图题
plt.title('Classifier BernoulliNB')
# 显示图片
plt.show()

