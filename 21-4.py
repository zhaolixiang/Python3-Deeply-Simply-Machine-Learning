# 导人波士顿房价数据集
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

boston = load_boston()
# 导人数据集拆分工具
from sklearn.model_selection import train_test_split

# 建立训练数据集和测试数据集
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
# 打印训练集和测试集的形态
# print(X_train.shape)
# print(X_test.shape)

# # 导人支持向量机回归模型
# from sklearn.svm import SVR
#
# # 分别测试linear 核函数和rbf 核函数
# for kernel in ['linear', 'rbf']:
#     svr = SVR(kernel=kernel)
#     svr.fit(X_train, y_train)
#     print(kernel, '核函数的模型训练集得分{:.3f}'.format(
#         svr.score(X_train, y_train)))
#     print(kernel, ' 核函数的模型测试集得分{:.3f} '.format(
#         svr.score(X_test, y_test)))


# 将每特征数值中的最小值和最大值用散点画出来
# plt.plot(X.min(axis=0), 'v', label='min')
# plt.plot(X.max(axis=0), '^', label='max')
# # 设定纵坐标为对数形式
# plt.yscale('log')
# # 设置图注位置为最佳
# plt.legend(loc='best')
# # 设定横纵轴标题
# plt.xlabel('features')
# plt.ylabel('feature magnitude')
# # 显示图形
# plt.show()


# 导人数据预处理工具
from sklearn.preprocessing import StandardScaler

# 对训练集和测试集进行数据预处理
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 将预处理后的数据特征最大值和最小值用散点图表示出来
plt.plot(X_train_scaled.min(axis=0), 'v', label='train set min ')
plt.plot(X_train_scaled.max(axis=0), '^', label='train set max ')
plt.plot(X_test_scaled.min(axis=0), 'v', label='train set min ')
plt.plot(X_test_scaled.max(axis=0), '^', label='train set max ')
plt.yscale('log')
# 设置图注位置
plt.legend(loc='best')
# 设置横纵轴标题
plt.xlabel('scaled features')
plt.ylabel('scaled feature magnitude')
# 显示图形
plt.show()
