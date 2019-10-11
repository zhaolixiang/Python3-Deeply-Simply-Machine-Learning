# 导人波士顿房价数据集
from sklearn.datasets import load_boston

boston = load_boston()
# 导人数据集拆分工具
from sklearn.model_selection import train_test_split

# 建立训练数据集和测试数据集
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
# 打印训练集和测试集的形态
# print(X_train.shape)
# print(X_test.shape)

# 导人支持向量机回归模型
from sklearn.svm import SVR

# 分别测试linear 核函数和rbf 核函数
for kernel in ['linear', 'rbf']:
    svr = SVR(kernel=kernel)
    svr.fit(X_train, y_train)
    print(kernel, '核函数的模型训练集得分{:.3f}'.format(
        svr.score(X_train, y_train)))
    print(kernel, ' 核函数的模型测试集得分{:.3f} '.format(
        svr.score(X_test, y_test)))
