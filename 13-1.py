# 导人nurnpy
import numpy as np

# 将x , y赋值为n p数组
X = np.array([[0, 1, 0, 1],
              [1, 1, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 1],
              [0, 1, 1, 0],
              [0, 1, 0, 1],
              [1, 0, 0, 1]])
y = np.array([0, 1, 1, 0, 1, 0, 0])
# 对不同分类计算每个特征为1 的数量
counts = {}
for label in np.unique(y):
    counts[label] = X[y == label].sum(axis=0)
# 打印计数结果
print("feature counts:\n{}".format(counts))
