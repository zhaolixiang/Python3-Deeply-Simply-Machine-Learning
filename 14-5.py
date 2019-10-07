# 导人多项式朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# 生成样本数量为500 ，分类数为5的数据集
X, y = make_blobs(n_samples=500, centers=5, random_state=8)
# 将数据集拆分成训练集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)

# 用多项式朴素贝叶斯拟合数据
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
mnb.score(X_test, y_test)
