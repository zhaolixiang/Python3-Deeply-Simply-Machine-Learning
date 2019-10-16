from sklearn.datasets import load_wine

# 从sklearn 的datasets 模块载入数据集
wine_dataset = load_wine()

# 打印酒数据集中的键
print("红酒数据集中的键：\n{}".format(wine_dataset.keys()))

# 使用.shape来打印数据的概况
print('数据概况：{}'.format(wine_dataset['data'].shape))

# 打印酒的数据集中的简短描述
# print(wine_dataset['DESCR'])

# 导人数据集拆分工具
from sklearn.model_selection import train_test_split

# 将数据集拆分为训练数据集和测试数据集
X_train, X_test, y_train, y_test = train_test_split(
    wine_dataset['data'],
    wine_dataset['target'],
    random_state=0)

# 打印训练数据集中特征向量的形态
print('X_train shape:{}'.format(X_train.shape))

# 打印测试数据集中的特征向量的形态
print('X_test shape:{}'.format(X_test.shape))

# 打印训练数据集中目标的形态
print('y_train shape{}'.format(y_train.shape))

# 打印测试数据集中目标的形态
print("y_test. shape{}".format(y_test.shape))

# 导人KNN分类模型

from sklearn.neighbors import KNeighborsClassifier

# 指定模型的n_neighbors参数值为1

knn = KNeighborsClassifier(n_neighbors=1)

# 用模型对数据进行拟合
knn.fit(X_train, y_train)
print(knn)


# 打印模型的得分
print("测试数据集得分{:.2f}".format(knn.score(X_test,y_test)))


import numpy as np
# 输入新的数据点
X_new=np.array([[13.2,2.77,2.51,18.5,96.6,1.04,2.55,0.57,1.47,6.2,1.05,3.33,820]])

# 使用. predict进行预测
prediction= knn .predict( X_new )
print('预测新红酒的分类为{}'.format(wine_dataset['target_names'][prediction]))








