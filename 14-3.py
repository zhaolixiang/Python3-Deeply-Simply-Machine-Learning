# 导入数据集生成工具
from sklearn.datasets import make_blobs
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split
# 导入高斯贝叶斯
from sklearn.naive_bayes import GaussianNB

# 生成样本数量为500 ，分类数为5的数据集
X, y = make_blobs(n_samples=500, centers=5, random_state=8)
# 将数据集拆分成训练集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

# 使用高斯贝叶斯拟合数据
gnb = GaussianNB()
gnb.fit(X_train, y_train)
# 打印模型得分
print('模型得分{:.3f}'.format(gnb.score(X_test, y_test)))
