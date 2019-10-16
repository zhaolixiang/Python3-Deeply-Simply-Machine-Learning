# 载入糖尿病情数据集
from sklearn.datasets import load_diabetes
# 导人岭回归
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, learning_curve, KFold
import numpy as np
import matplotlib.pylab as plt

X, y = load_diabetes().data, load_diabetes().target


# 定义一个绘制学习曲线的函数
def plot_learning_curve(est, X, y):
    # 将数掘进行20次拆分用来对模型进行评分
    training_set_size, train_scores, test_scores = learning_curve(
        est, X, y, train_sizes=np.linspace(.1, 1, 20),
        cv=KFold(20, shuffle=True, random_state=1))
    estimator_name = est.__class__.__name__
    line = plt.plot(training_set_size, train_scores.mean(axis=1), '--', label='training' + estimator_name)
    plt.plot(training_set_size, test_scores.mean(axis=1), '-', label='test' + estimator_name,
             c=line[0].get_color())
    plt.xlabel('Training set size')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)


plot_learning_curve(Ridge(alpha=1), X, y)
plot_learning_curve(LinearRegression(), X, y)
plt.legend(loc=(0, 1.05), ncol=2, fontsize=11)
plt.show()
