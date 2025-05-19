import pandas as pd

# 读取CSV文件
df = pd.read_csv('E:\学习\毕业论文\\data\\problem_actions_session_1_test_2.csv',skiprows=0)

# 统计总行数
total_rows = len(df)

# 统计不同的CaseID数量
unique_caseid = df['enroll_id'].nunique()

# 统计不同的ActivityID数量
unique_activityid = df['object'].nunique()

# 统计CaseID的最大重复次数
max_repeat = df['enroll_id'].value_counts().max()

# 统计CaseID的最小重复次数
min_repeat = df['enroll_id'].value_counts().min()

print("轨迹数量:", unique_caseid)
print("事件数量:", total_rows)
print("活动数量:", unique_activityid)
print("轨迹最大长度:", max_repeat)
print("轨迹最小长度:", min_repeat)