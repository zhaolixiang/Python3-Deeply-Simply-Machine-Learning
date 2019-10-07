# 导入线性回归模型
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pylab as plt

# 生成用于回归分析的数据集

X, y = make_regression(n_samples=50, n_features=1, n_informative=1, noise=50, random_state=1)

# 使用线性模型对数据进行拟合
reg = LinearRegression()
reg.fit(X, y)

# z是我们生成的等差数列，用来画出线性模型的图形

z = np.linspace(-3, 3, 200).reshape(-1, 1)

plt.scatter(X, y, c='b', s=60)
plt.plot(z, reg.predict(z), c='k')

plt.title('Linear Regression')
plt.show()



# 打印直线的系数和截距
print('直线的系数是：{:.2f}'.format(reg.coef_[0]))
print('直线的截距是：{:.2f}'.format(reg.intercept_))


