# 载入糖尿病情数据集
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = load_diabetes().data, load_diabetes().target

# 将数据集拆分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
# 使用线性回归模型进行拟合
lr = LinearRegression().fit(X_train, y_train)

print("训练数据集得分:{:.2f}".format(lr.score(X_train, y_train)))
print("测试数据集得分:{:.2f}".format(lr.score(X_test, y_test)))
