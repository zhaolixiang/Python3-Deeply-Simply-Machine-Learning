import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt
# 导人数据集获取工具
from sklearn.datasets import fetch_lfw_people

# 载入人脸数据集
faces = fetch_lfw_people(min_faces_per_person=20, resize=0.8)
image_shape = faces.images[0].shape

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# 对数据集进行拆分
X_train, X_test, y_train, y_test = train_test_split(faces.data / 255, faces.target, random_state=62)

# 使用自化功能处理人脸数据
pca = PCA(whiten=True, n_components=0.9, random_state=62).fit(X_train)
X_train_whiten = pca.transform(X_train)
X_test_whiten = pca.transform(X_test)
# 打印自化后数据形态
print('白化后数据形态：{}'.format(X_train_whiten.shape))

# 导人神经网络
from sklearn.neural_network import MLPClassifier

# 训练神经网络
mlp = MLPClassifier(hidden_layer_sizes=[100, 100], random_state=62, max_iter=400)
# 使用自化后的数据训练神经网络
mlp.fit(X_train_whiten, y_train)
# 打印模型准确率
print('数据白化后模型识到准确率:{:.2f}'.format(mlp.score(X_test_whiten, y_test)))
