# 导入numpy
import numpy as np
# 导人画图工具
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# 导入tree 模型和数据集加载工具
from sklearn import tree, datasets
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split

wine = datasets.load_wine()
# 只选取数据集的前两个特征
X = wine.data[:, : 2]
y = wine.target
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 设定决策树分类器最大深度为3
clf = tree.DecisionTreeClassifier(max_depth=3)
# 拟合训练数据集
clf.fit(X_train, y_train)

# 导入graphviz工具
import graphviz
# 导入决策树中输出graphviz的接口
from sklearn.tree import export_graphviz

# 选择最大深度为3的分类模型
export_graphviz(clf, out_file="wine.dot", class_names=wine.target_names,
                feature_names=wine.feature_names[: 2], impurity=False, filled=True)
# 打开一个dot文件
with open("wine.dot") as f:
    dot_graph = f.read()
# 显示dot文件中的图形
dot=graphviz.Source(dot_graph)
dot.view()
