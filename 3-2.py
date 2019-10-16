import numpy as np
from scipy import sparse

matrix = np.eye(6)
# 上面用numpy的eye 函数生成一个6行6列的对角矩阵
# 矩阵中对角线上的元素数值为1 ，其余都是0

sparse_matrix=sparse.csr_matrix(matrix)
# 这一行把np数组转化为CSR格式的Scipy稀疏矩阵（sparse matrix)
#sparse 函数只会存储非0元素

# 将生成的对角矩阵打印出来
print("对角矩阵:\n{}".format(matrix))

# 将sparse 函数生成的矩阵打印出来进行对比
print("\n sparse 存储的矩阵:\n{}".format(sparse_matrix))
