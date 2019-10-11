# 导入数据集获取工具
from sklearn.datasets import fetch_mldata

# 加载MNIST 手写数字数据集

mnist = fetch_mldata('MNIST original')
print(mnist)

# 打印样本数量和样本特征数
print("样本数量:{},样本特征数{}".format(mnist.data.shape[0],
                               mnist.data.shape[1]))

# 导人数据集拆分工具
from sklearn.model_selection import train_test_split

# 建立训练数据集和测试数据集
X = mnist.data / 255.
y = mnist.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=5000, test_size=1000, random_state=62)

from sklearn.neural_network import MLPClassifier

# 设置神经网络有两个100 个节点的隐藏层
mlp_hw = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[100, 100],
                       activation='relu', alpha=1e-5, random_state=62)
# 使用数据训练神经网络模型
mlp_hw.fit(X_train, y_train)
# 打印模型分数
print('测试数据集得分:{:.2f}%'.format(mlp_hw.score(X_test, y_test) * 100))

# 导入图像处理工具
from PIL import Image
import numpy as np

# 打开图像
image = Image.open('4.jpg').convert('F')
# 调整图像的大小
image = image.resize((28, 28))
arr = []
# 将图像中的像素作为预测数据点的特征
for i in range(28):
    for j in range(28):
        pixel = 1.0 - float(image.getpixel((j, i))) / 255.
        arr.append(pixel)

# 由子只有一个样本， 所以需要进行reshape操作
arr1 = np.array(arr).reshape(1, -1)
# 进行图像识别
print('图片中的数字是:{:.0f}'.format(mlp_hw.predict(arr1)[0]))
