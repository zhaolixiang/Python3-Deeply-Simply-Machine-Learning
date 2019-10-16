# 导人波士顿房价数据集
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

boston = load_boston()
# 导人数据集拆分工具
from sklearn.model_selection import train_test_split

# 建立训练数据集和测试数据集
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

# 导人支持向量机回归模型
from sklearn.svm import SVR
# 导人数据预处理工具
from sklearn.preprocessing import StandardScaler

# 对训练集和测试集进行数据预处理
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 用预处理后的数据重新训练模型
for kernel in ['linear', 'rbf']:
    svr = SVR(kernel=kernel,C=100,gamma=0.1)
    svr.fit(X_train_scaled, y_train)
    print('数据预处理后', kernel, '核函数的模型训练集得分{:.3f}'.format(
        svr.score(X_train_scaled, y_train)))
    print('数据预处理后', kernel, ' 核函数的模型测试集得分{:.3f} '.format(
        svr.score(X_test_scaled, y_test)))

