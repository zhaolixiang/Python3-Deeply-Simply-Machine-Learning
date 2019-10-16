# 导人红酒数据集
from sklearn.datasets import load_wine
# 导入MLP神经网络
from sklearn.neural_network import MLPClassifier
# 导人数据集拆分工具
from sklearn.model_selection import train_test_split

# 建立训练集和测试集


wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target,
                                                    random_state=62)
# 打印数据形态
print(X_train.shape, X_test.shape)

# 设定MLP神经网络的参数
mlp = MLPClassifier(hidden_layer_sizes=[100, 100], max_iter=400,
                    random_state=62)
# 使用MLP拟合数据
mlp.fit(X_train, y_train)
# 打印模型得分
print('模型得分：{:.2f}'.format(mlp.score(X_test, y_test)))

from sklearn.preprocessing import MinMaxScaler

# 使用MinMaxScaler进行数据预处理
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_pp = scaler.transform(X_train)
X_test_pp = scaler.transform(X_test)
# 重新训练模型
mlp.fit(X_train_pp, y_train)
# 打印模型分数
print("数据预处理后的模型得分:{:.2f}".format(mlp.score(X_test_pp, y_test)))
