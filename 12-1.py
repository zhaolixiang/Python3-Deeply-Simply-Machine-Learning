# 载入糖尿病情数据集
from sklearn.datasets import load_diabetes
# 导人套索回归
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
# 使用套索回归拟合数据
lasso = Lasso().fit(X_train, y_train)
print("套索回归在训练数据集的得分{:.2f}".format(lasso.score(X_train, y_train)))
print("套索回归在测试数据集的得分{:.2f}".format(lasso.score(X_test, y_test)))
print("套索回归使用的特征数{}".format(np.sum(lasso.coef_ != 0)))
