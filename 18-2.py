# 导人pandas 库
import pandas as pd

# 用pandas 打开csv 文件
data=pd.read_csv('18-1.csv',header=None,index_col=False,
                 names=['年龄','单位性质','权重','学历','受教育时长',
                        '婚姻状况','职业','家庭情况'
                     , '种族','性别','资产所得','资产损失','周工作时长','原籍','收入'])

# 为了方便展示， 我们选取其中一部分数据
data_lite=data[['年龄','单位性质','权重','学历','受教育时长','职业','收入']]

# 使用get_dummies 将文本数据转化为数值
data_dummies=pd.get_dummies(data_lite)
# 对比样本原始特征和虚拟变最特征
print('样本原始特征：\n',list(data_lite.columns),'\n')
print('虚拟变量特征：\n',list(data_dummies.columns))



