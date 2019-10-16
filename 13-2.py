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

# 要进行预测的这一天，没有刮北风，也不闷热
# 但是多云，天气预报没有说有雨
Next_Day = [[0, 0, 1, 0]]
pre = elf.predict(Next_Day)
if pre == [1]:
    print("要下雨啦，快收衣服啊！")
else:
    print("放心,又是一个艳阳天")
