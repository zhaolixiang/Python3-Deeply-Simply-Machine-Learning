# 载入糖尿病情数据集
from sklearn.datasets import load_diabetes
# 导人岭回归
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
# 使用线性回归模型进行拟合
lr = LinearRegression().fit(X_train, y_train)
# 使用岭回归对数据进行拟合
ridge = Ridge(alpha=1).fit(X_train, y_train)
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("岭回归的训练数据集得分:{:.2f} ".format(ridge.score(X_train, y_train)))
print("岭回归的测试数据集得分:{:.2f} ".format(ridge.score(X_test, y_test)))

# 绘制alpha=1 时的模型系数
plt.plot(ridge.coef_, 's', label='Ridge alpha=1')
# 绘制alpha=lO 时的模型系数
plt.plot(ridge10.coef_, '^', label='Ridge alpha=10')
# 绘制alpha=O . 1时的模型系数
plt.plot(ridge01.coef_, 'v', label='Ridge alpha=0.1')
# 绘制线性回归的系数作为对比
plt.plot(lr.coef_, 'o', label='Linear regression')
plt.xlabel('coefficient index')
plt.ylabel('coefficient magnitude')
plt.hlines(0,0,len(lr.coef_))
plt.legend()
plt.show()
