# 导人威斯康星乳腺肿瘤数据集
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

cancer = load_breast_cancer()

# 将数据集的数值和分类目标赋值给x手的
X, y = cancer.data, cancer.target
# 使用数据集拆分工具拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38)

# 使用高斯朴素贝叶斯拟合数据
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 打印模型预测的分类和真实的分类
print('模型预测的分类是{}'.format(gnb.predict([X[312]])))
print('样本的正确分类是', y[312])
