import matplotlib.pyplot as plt
# 导入dendrogram和ward工具
from scipy.cluster.hierarchy import dendrogram, ward
# 导入数据集生成工具
from sklearn.datasets import make_blobs

# 生成分类数为1 的数据集
blobs = make_blobs(random_state=1, centers=1)
X_blobs = blobs[0]

# 使用连线的方式进行可视化
linkage = ward(X_blobs)
dendrogram(linkage)
ax = plt.gca()
# ax.add_patch()
# 设定横纵轴标签
plt.xlabel('Sample index')
plt.ylabel('Cluster distance')
# 显示图像
plt.show()


