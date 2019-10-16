import ssl


ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt
# 导人数据集获取工具
from sklearn.datasets import fetch_lfw_people

# 载入人脸数据集
faces = fetch_lfw_people(min_faces_per_person=20, resize=0.8)
image_shape = faces.images[0].shape
# 将照片打印出来
# fig, axes= plt.subplots(3,4,figsize=(l2 , 9),
# subplot kw=(’ xticks ’ :(),’ yti cks ’:() })
fig, axes = plt.subplots(3, 4, figsize=(12, 9), subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(faces.target, faces.images, axes.ravel()):
    ax.imshow(image, cmap=plt.cm.gray)
    ax.set_title(faces.target_names[target])

# 显示图像
plt.show()


from sklearn.model_selection import train_test_split
# 导人神经网络
from sklearn.neural_network import  MLPClassifier
# 对数据集进行拆分
X_train,X_test,y_train,y_test=train_test_split(faces.data/255,faces.target,random_state=62)
# 训练神经网络
mlp=MLPClassifier(hidden_layer_sizes=[100,100],random_state=62,max_iter=400)
mlp.fit(X_train,y_train)
# 打印模型准确率
print('模型识别准确率:{:.2f}'.format(mlp.score(X_test,y_test)))
