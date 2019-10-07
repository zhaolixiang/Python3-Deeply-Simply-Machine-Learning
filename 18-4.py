# 导人pandas 库
import pandas as pd
# 导入tree 模型和数据集加载工具
from sklearn import tree, datasets
# 用pandas 打开csv 文件
from sklearn.model_selection import train_test_split

data = pd.read_csv('18-1.csv', header=None, index_col=False,
                   names=['年龄', '单位性质', '权重', '学历', '受教育时长',
                          '婚姻状况', '职业', '家庭情况'
                       , '种族', '性别', '资产所得', '资产损失', '周工作时长', '原籍', '收入'])

# 为了方便展示， 我们选取其中一部分数据
data_lite = data[['年龄', '单位性质', '权重', '学历', '受教育时长', '职业', '收入']]

# 使用get_dummies 将文本数据转化为数值
data_dummies = pd.get_dummies(data_lite)

# 定义数据集的特征值
features = data_dummies.loc[:, '年龄':'职业_ Transport-moving']
# 将特征数值赋值为x
X = features.values
# 将收入大于5Ok作为预测目标
y = data_dummies['收入_ >50K'].values

print('特征形态：{}, 标签形态:{}'.format(X.shape, y.shape))

# 将数据及拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 用最大深度为5的随机森林拟合数据
go_dating_tree = tree.DecisionTreeClassifier(max_depth=5)
go_dating_tree.fit(X_train, y_train)
print("模型得分：{:.2f}".format(go_dating_tree.score(X_test, y_test)))

# 将Mr z的数据输入给模型
MrZ = [[37, 40, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# 使用模型做出预测
dating_dec = go_dating_tree.predict(MrZ)
if dating_dec == 1:
    print("大胆去追求真爱吧，这哥们月薪过5万了")
else:
    print("不用去了，不满足你的要求")
