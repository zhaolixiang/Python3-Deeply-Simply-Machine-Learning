# 导人威斯康星乳腺肿瘤数据集
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

# 打印数据集键值
print(cancer.keys())


print('肿瘤的分类', cancer['target_names'])
print('肿瘤的特征', cancer['feature_names'])
