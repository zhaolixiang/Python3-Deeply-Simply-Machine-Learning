# 导入线性回归模型
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pylab as plt

# 输入两个点的横坐标
X = [[1], [4]]

# 输入两个点的纵坐标
y = [3, 5]

# 用线性模型拟合这个两个点
lr = LinearRegression().fit(X, y)

# 画出两个点和直线的图形
z = np.linspace(0, 5, 20)
plt.scatter(X, y, s=80)
plt.plot(z, lr.predict(z.reshape(- 1, 1)), c='k')

# 设定图题为Straight Line
plt.title('Straight Line')

# 将图片显示出来
plt.show()

print("直线方程为：")
print('y={:.3f}'.format(lr.coef_[0]),'x','+{:.3f}'.format(lr.intercept_))
