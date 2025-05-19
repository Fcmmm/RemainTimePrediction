import pandas as pd
from datetime import datetime
# 读取CSV文件
# data = pd.read_csv('E:\学习\毕业论文\\data\\video_actions_session_1.csv',skiprows=0)
# 将结果写入新的CSV文件
# df.to_csv('E:\学习\毕业论文\\data\\video_actions_session_1_test3.csv', index=False)

# 统计不同的enroll_id数量
# enroll_id_count = data['enroll_id'].nunique()
# 统计不同的session_id数量
# session_id_count = data['session_id'].nunique()
# print(f"不同的enroll_id数量：{enroll_id_count}")
# print(f"不同的session_id数量：{session_id_count}")

# 统计每个enroll_id和object的出现次数
# counts = df.groupby(['enroll_id', 'object']).size().reset_index(name='count')
# 找出同一个enroll_id下的同一个object只出现了一次的情况
# unique_objects = counts[counts['count'] == 1]
# 筛选出符合条件的enroll_id
# unique_enroll_ids = unique_objects['enroll_id']
# 根据筛选出的enroll_id提取数据
# filtered_df = df[df['enroll_id'].isin(unique_enroll_ids)]

# 统计每个enroll_id下每个object的数量
# df_count = df.groupby(['enroll_id', 'object']).size().reset_index(name='count')
# 获取每个enroll_id下的object数量小于10的enroll_id
# enroll_ids_less_than_10 = df_count.groupby('enroll_id').size()[df_count.groupby('enroll_id').size() == 5].index
# print(len(enroll_ids_less_than_10))
# 筛选出符合条件的enroll_id的所有信息
# filtered_df = df[df['enroll_id'].isin(enroll_ids_less_than_10)]
# 将筛选出的数据写入新的CSV文件
# filtered_df.to_csv('E:\学习\毕业论文\\data\\video_actions_session_1_test3.csv', index=False)

# 读入原始CSV文件
df = pd.read_csv('E:\学习\毕业论文\\data\\video_actions_session_1_test3.csv',skiprows=0)

# 将时间戳转换为 datetime 对象
df['Time'] = pd.to_datetime(df['Time'])

# 对数据按照 enroll_id 进行排序
df = df.sort_values(by=['enroll_id', 'Time']).reset_index(drop=True)

# 计算每两行之间的时间间隔
df['time_diff'] = df.groupby('enroll_id')['Time'].diff()

# 判断相邻两行的enroll_id是否相同
df['same_enroll_id'] = df['enroll_id'] == df['enroll_id'].shift(1)

# 筛选时间间隔超过1小时且相邻两行的enroll_id相同的行
df_time_diff_gt_1_hour = df[(df['time_diff'] > pd.Timedelta(hours=1)) & (df['enroll_id'] == df['enroll_id'].shift(1))]

# 输出结果
print(df_time_diff_gt_1_hour[['enroll_id', 'Time', 'time_diff']])

unique_enroll_ids = df_time_diff_gt_1_hour['enroll_id'].unique()
# 根据筛选出的enroll_id提取数据
filtered_df = df[df['enroll_id'].isin(unique_enroll_ids)]
filtered_df.to_csv('E:\学习\毕业论文\\data\\video_actions_session_1_test4.csv', index=False)