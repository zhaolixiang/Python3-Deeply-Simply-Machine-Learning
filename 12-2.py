# 载入糖尿病情数据集
from sklearn.datasets import load_diabetes
# 导人套索回归
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

# 增加最大迭代次数的默认设置
# 否则模型会提示我们增加最大迭代次数
lassoO1 = Lasso(alpha=0.1, max_iter=100000).fit(X_train, y_train)
print("alpha=O.1 时套索回归在训练数据集的得分{:.2f}".format(lassoO1.score(X_train, y_train)))
print("alpha=O.1 时套索回归在测试数据集的得分{:.2f}".format(lassoO1.score(X_test, y_test)))
print("alpha=O.1 时套索回归使用的特征数{}".format(np.sum(lassoO1.coef_ != 0)))
