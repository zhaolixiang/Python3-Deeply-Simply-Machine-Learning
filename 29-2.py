# 导人numpy
import numpy as np
# 导入画图工具
import matplotlib.pyplot as plt

# 生成随机数列
rnd = np.random.RandomState(38)
x = rnd.uniform(-5, 5, size=50)
# 向数据中添加噪声
y_no_noise = (np.cos(6 * x) + x)
X = x.reshape(-1, 1)
y = (y_no_noise + rnd.normal(size=len(x))) / 2
# 绘制图形
# plt .plot (X, y ,’o ’, c = ’ E ’ )
plt.plot(X, y, 'o', c='r')
# 显示图形
plt.show()

# 导入神经网络
from sklearn.neural_network import MLPRegressor
# 导人KNN
from sklearn.neighbors import KNeighborsRegressor

# 生成一个等差数列
line = np.linspace(- 5, 5, 1000, endpoint=False).reshape(-1, 1)
# 分别用两种算法拟合数据
mlpr = MLPRegressor().fit(X, y)
knr = KNeighborsRegressor().fit(X, y)
# 绘制图形
plt.plot(line, mlpr.predict(line), label='MLP')
plt.plot(line, knr.predict(line), label='KNN')
plt.plot(X, y, 'o', c='r')
plt.legend(loc='best')
# 显示图形
plt.show()

# 设置箱体数为11
bins=np.linspace(-5,5,11)
# 将数据进行装箱操作
target_bin=np.digitize(X,bins=bins)
# 打印装箱数据范围
print('装箱数据范围:{}'.format(bins))
# 打印前十个数据的特征值
print('前十个数据点的特征值:{}'.format(X[:10]))
# 找到它们所在的箱子
print('前十个数据点所在的箱子:{}'.format(target_bin[:10]))


# 导入独热编码
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(sparse = False)
onehot . fit ( target_bin )

# 使用独热编码转化数据
X_in_bin = onehot . transform(target_bin)
# 打印结果
# print （ ’ 装箱后的数据形态｛｝ ’ . format(X_in一bin . shape))
# print （’ ＼ n 装箱后的前十个数据点。＼ n ｛） ’. forma t ( X_in_bin [ : 1 0) ))
print('装箱后的数据形态:{}'.format(X_in_bin.shape))


