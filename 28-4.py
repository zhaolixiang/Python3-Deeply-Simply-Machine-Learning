# 导入DBSCAN
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
# 导入数据集生成工具
from sklearn.datasets import make_blobs

# 生成分类数为1 的数据集
blobs = make_blobs(random_state=1, centers=1)
X_blobs = blobs[0]

db = DBSCAN()
# 使用DBSCAN 拟合数据
clusters = db.fit_predict(X_blobs)
# 绘制散点图
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=clusters, cmap=plt.cm.cool, s=60, edgecolors='k')
# 设置横纵轴标签
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
# 显示图像
plt.show()

# 打印聚类个数
print('聚类标签为：{}'.format(clusters))
