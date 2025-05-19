# -*- coding = UTF-8 -*-
# @Time : 2024/10/1 23:45
# @Author : 付传萌
# @File : summ.py
# @Software : PyCharm
import pandas as pd

# 读取CSV文件
df = pd.read_csv('data/codeforces 948_3.csv')

# 假设要去除的列名为 'column_name'
# 去除重复值并创建一个新的DataFrame
unique_values = df['who'].drop_duplicates()

# 统计去除重复值后的数量
count_unique = unique_values.count()

print("去除重复值后的数据：")
print(unique_values)
print(f"去除重复值后的数量: {count_unique}")
