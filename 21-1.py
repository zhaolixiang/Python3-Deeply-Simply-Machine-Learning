# 导人波士顿房价数据集
from sklearn.datasets import load_boston

boston = load_boston()
# 打印数据集中的键
print(boston.keys())

# 打印数据集中的短描述
print(boston['DESCR'])
