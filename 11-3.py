# 载入糖尿病情数据集
from sklearn.datasets import load_diabetes
# 导人岭回归
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
# 使用岭回归对数据进行拟合
ridge = Ridge(alpha=0.1).fit(X_train, y_train)
print("岭回归的训练数据集得分:{:.2f} ".format(ridge.score(X_train, y_train)))
print("岭回归的测试数据集得分:{:.2f} ".format(ridge.score(X_test, y_test)))
