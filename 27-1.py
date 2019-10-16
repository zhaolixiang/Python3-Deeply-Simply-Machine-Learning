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
