import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt
# 导人数据集获取工具
from sklearn.datasets import fetch_lfw_people

# 载入人脸数据集
faces = fetch_lfw_people(min_faces_per_person=20, resize=0.8)
image_shape = faces.images[0].shape

# 导入NMF
from sklearn.decomposition import NMF

from sklearn.model_selection import train_test_split

# 对数据集进行拆分
X_train, X_test, y_train, y_test = train_test_split(faces.data / 255, faces.target, random_state=62)

# 使用NMF处理数据
nmf = NMF(n_components=105, random_state=62).fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)
# 打印NMF处理后的数据形态
print('NMF处理后数据形态:{}'.format(X_train_nmf.shape))

# 导人神经网络
from sklearn.neural_network import MLPClassifier

# 训练神经网络
mlp = MLPClassifier(hidden_layer_sizes=[100, 100], random_state=62, max_iter=400)
# 用NMF处理后的数据训练网络
mlp.fit(X_train_nmf, y_train)
# 打印模型准确率
print('NMF 处理后模型准确率:{:.2f}'.format(mlp.score(X_test_nmf, y_test)))
