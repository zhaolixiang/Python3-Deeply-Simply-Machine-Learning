# 载入糖尿病情数据集
from sklearn.datasets import load_diabetes
# 导人套索回归
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pylab as plt

X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
lasso = Lasso(alpha=1, max_iter=100000).fit(X_train, y_train)

# 增加最大迭代次数的默认设置
# 否则模型会提示我们增加最大迭代次数
lasso01 = Lasso(alpha=0.1, max_iter=100000).fit(X_train, y_train)
lasso0001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)

# 使用岭回归对数据进行拟合
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

# 绘制alpha值等于1时的模型系数
plt.plot(lasso.coef_, 's', label="Lasso alpha1`")
# 绘制alpha值等于0.1时的模型系数
plt.plot(lasso01.coef_, '^', label="Lasso alpha=0.1")
# 绘制alpha值等于0.0001 时的模型系数
plt.plot(lasso0001.coef_, 'v', label="Lasso alpha=0.001")
# 绘制） alpha 值等于0 . 1 时的岭回归模型系数作为对比
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-1000, 800)
plt.xlabel(" Coefficient index")
plt.show()
