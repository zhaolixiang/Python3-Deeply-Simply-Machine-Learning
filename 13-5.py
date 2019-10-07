# 导人nurnpy
import numpy as np
# 导人贝努利贝叶斯
from sklearn.naive_bayes import BernoulliNB

# 将x , y赋值为n p数组
X = np.array([[0, 1, 0, 1],
              [1, 1, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 1],
              [0, 1, 1, 0],
              [0, 1, 0, 1],
              [1, 0, 0, 1]])
y = np.array([0, 1, 1, 0, 1, 0, 0])
# 使用贝努利贝叶斯拟合数据
elf = BernoulliNB()
elf.fit(X, y)

# 假设另外一天的数据如下
Another_day = [[1,1,0,1]]
# 模型预测分类的概率
print(elf.predict_proba(Another_day))
