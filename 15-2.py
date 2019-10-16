# 导人威斯康星乳腺肿瘤数据集
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

# 将数据集的数值和分类目标赋值给x手的
X, y = cancer.data, cancer.target
# 使用数据集拆分工具拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38)
# 打印训练集和测试集的数据形态
print("训练集数据形态:", X_train.shape)
print("测试集数据形态:", X_test.shape)
