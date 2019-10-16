# 导人数据集拆分工具
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=2, n_informative=2, random_state=38)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_:{}".format(lr.coef_[:]))
print("lr.intercept:{}".format(lr.intercept_))


print('训练数据集得分:{:.2f}'.format(lr.score(X_train,y_train)))
print('测试数据集得分:{:.2f}'.format(lr.score(X_test,y_test)))





