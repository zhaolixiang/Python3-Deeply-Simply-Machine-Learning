import matplotlib.pyplot as plt
# 导入数据集生成工具
from sklearn.datasets import make_blobs
import numpy as np
# 导入KMeans 工具
from sklearn.cluster import KMeans

# 生成分类数为1 的数据集
blobs = make_blobs(random_state=1, centers=1)
X_blobs = blobs[0]



# ＃要求KMeans 将数据聚为3 类
kmeans = KMeans(n_clusters=3)
# 拟合数据
kmeans.fit(X_blobs)
# 下面是用来画图的代码
x_min, x_max = X_blobs[:, 0].min() - 0.5, X_blobs[:, 0].max() + 0.5
y_min, y_max = X_blobs[:, 1].min() - 0.5, X_blobs[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.summer, aspect='auto', origin='lower')
plt.plot(X_blobs[:, 0], X_blobs[:, 1], 'r.', markersize=5)
# 用蓝色叉号代表聚类的中心
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=3, color='b', zorder=10)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
# 显示图像
plt.show()


# 打印KMeansi进行聚类的标签
print('K均值的聚类标签:{}'.format(kmeans.labels_))
