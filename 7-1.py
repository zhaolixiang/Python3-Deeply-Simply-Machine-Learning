import matplotlib.pylab as plt
# 导入make_regression数据集生成器
from sklearn.datasets import make_regression

# 生成特征数量为1 ， 噪音为50的数据集
X, y = make_regression(n_features=1, n_informative=1, noise=50, random_state=8)

# 用散点图将数据点进行可视化
plt.scatter(X, y, c='orange', edgecolors='k')
plt.show()

import numpy as np
# 导人用于回归分析的KNN模型
from sklearn.neighbors import KNeighborsRegressor

reg = KNeighborsRegressor()
# 用KNN 模型拟合数据
reg.fit(X, y)
# 把预测结果用图像进行可视化
z = np.linspace(-3, 3, 200).reshape(-1, 1)
plt.scatter(X, y, c='orange', edgecolors='k')
plt.plot(z, reg.predict(z), c='k', linewidth=3)
# 向图像添加标题
plt.title('KNN Regressor')
plt.show()

print('模型评分:{:.2f}'.format(reg.score(X, y)))

from sklearn.neighbors import KNeighborsRegressor

# 减少模型的n_neighbors参数为2

reg2 = KNeighborsRegressor(n_neighbors=2)

reg2.fit(X, y)

# 重新进行可视化
plt.scatter(X, y, c='orange', edgecolor='k')
plt.plot(z, reg2.predict(z), c='k', linewidth=3)
plt.title('KNN Regressor')
plt.show()

print('模型评分:{:.2f}'.format(reg2.score(X, y)))
