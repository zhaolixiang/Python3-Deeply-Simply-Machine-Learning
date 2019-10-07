import pandas


# 先创建一个同学个人信息的小数据集

data = {
    "Name":["小芋","小菌","小榆","小检"],
    "City":["北京","上海","广州","深圳"],
    "Age":["18","20","22","24"],
    "Height":["162","161","165","166"],
}

data_frame=pandas.DataFrame(data)
display(data_frame)

# 显示所有不在北京的同学信息
display(data_frame[data_frame.City!="北京"])
