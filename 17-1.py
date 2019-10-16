# 导入随机森林模型
from sklearn.ensemble import RandomForestClassifier
# 导入tree 模型和数据集加载工具
from sklearn import tree, datasets
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split
# 载入红酒数据集
wine=datasets.load_wine()
# 选择数据集前两个特征
X=wine.data[:,:2]
y=wine.target

# 将数据集拆分为训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y)

# 设定随机森林中有6棵树
forest=RandomForestClassifier(n_estimators=6,random_state=3)
# 使用模型拟合数据
forest.fit(X_train,y_train)
print(forest)