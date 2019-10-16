import matplotlib.pyplot as plt
# 导入数据集生成工具
from sklearn.datasets import make_blobs

# 生成分类数为1 的数据集
blobs = make_blobs(random_state=1, centers=1)
X_blobs = blobs[0]
# 绘制散点图
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c='r', edgecolors='k')
# 显示图像
plt.show()
