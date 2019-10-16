# 导入numpy
import numpy as np
# 导入画图工具
import matplotlib.pyplot as plt

# 生成一个等差数列
line = np.linspace(-5, 5, 200)

# 画出非线性矫正的图形表示
plt.plot(line,np.tanh(line),label='tanh')
plt.plot(line,np.maximum(line,0),label='relu')

# 设置图注位置
plt.legend(loc='best')
# 设置横纵轴标题
plt.xlabel('x')
plt.ylabel('relu(x) and tanh(x)')
# 显示图形
plt.show()
