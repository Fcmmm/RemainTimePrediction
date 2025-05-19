from tkinter import Variable
import numpy as np
import torch
import networkx as nx
from itertools import combinations, chain
from input_data import InputData

def getpredict(list,m):
    for i in range(64):
        if (len(list) < 64):
            list.append(list[0])
        else:
            break
    list = np.array(list)
    list = torch.LongTensor(list)

    if m == 1:
        prediction = model(list)
    else:
        prediction = model2(list)
    return prediction[0].item()

def detect(model, input_batchs):
    predict_list = list()
    target_list = list()
    input_list = list()
    for (input,target) in input_batchs:
        # for i in range(len(input)):
        #    print(data.id2event[input[i][0]],data.id2event[input[i][1]])
        input = np.array(input)
        input = torch.LongTensor(input)

        prediction = model(input)
        predict_list += [pdic.item() for pdic in prediction]
        target_list += target
        input_list += input

    for i in range(0,len(predict_list)):
        print(data.id2event[input_list[i].numpy().tolist()[0]][0],data.id2event[input_list[i].numpy().tolist()[1]][0], end="")
        print(predict_list[i], end="")
        print(' ', end="")
        print(target_list[i])
    MAE = computeMAE(target_list, predict_list)
    print(MAE)

def computeMAE(list_a,list_b):
    MAE_temp = []
    for num in range(len(list_a)):
        MAE_temp.append(abs(list_a[num]-list_b[num]))
    MAE = sum(MAE_temp)/len(list_a)
    return MAE

# 创建一个无向图
G = nx.DiGraph()

# 添加节点
topics = ["A", "B", "C", "D", "E", "F"]
# topics = ["greedy implementation sortings", "greedy DP math", "greedy implementation DP binary_search trees", "greedy implementation bitmasks brute_force constructive_algorithms", "DP math number_theory combinatorics", "DP DFS data_structures 2-set"]
G.add_nodes_from(topics, type="")

# 添加知识点节点
knowledge_points = ["greedy", "implementation", "sortings", "dp", "math", "binary search", "trees",
                    "bitmasks", "brute force", "constructive algorithms", "combinatorics", "number theory",
                    "2-set", "data structures", "dfs"]
G.add_nodes_from(knowledge_points, type="知识点")

# 添加与知识点之间的连接关系
relations = {
    "A": ["greedy", "implementation", "sortings"],
    "B": ["greedy", "dp", "math"],
    "C": ["greedy", "implementation", "dp", "binary search", "trees"],
    "D": ["greedy", "implementation", "bitmasks", "brute force", "constructive algorithms"],
    "E": ["dp", "math", "number theory", "combinatorics"],
    "F": ["dp", "dfs", "data structures", "2-set"]
}

# relations = {
#     "greedy implementation sortings": ["greedy", "implementation", "sortings"],
#     "greedy DP math": ["greedy", "dp", "math"],
#     "greedy implementation DP binary_search trees": ["greedy", "implementation", "dp", "binary search", "trees"],
#     "greedy implementation bitmasks brute_force constructive_algorithms": ["greedy", "implementation", "bitmasks", "brute force", "constructive algorithms"],
#     "DP math number_theory combinatorics": ["dp", "math", "number theory", "combinatorics"],
#     "DP DFS data_structures 2-set": ["dp", "dfs", "data structures", "2-set"]
# }

for topic, related_knowledge_points in relations.items():
    for kp in related_knowledge_points:
        G.add_edge(topic, kp)

# 使用拓扑排序来确定题目的顺序
topic_order = list(nx.topological_sort(G))[0:6]

# 前缀长度
prefix_length = 5

# model = torch.load('./model/codeforces 936_13/LSTM/embdDim3_lossL1Loss_optimAdam_hiddenDim5_startPos1_trainTypemix_nLayer1_dropout1/2024-05-08(09-54-46).pth')#TODO 参数传递pth文件路径
#
# model2 = torch.load('E:\学习\毕业论文\data\可用模型\codeforces 936_14 后缀用时.pth')
#  前缀预测
model = torch.load('model\\codeforces 936_16\\LSTMAtt\\embdDim3_lossL1Loss_optimAdam_hiddenDim5_startPos1_trainTypeiteration_nLayer1_dropout1\\2024-11-09(16-19-42).pth')#TODO 参数传递pth文件路径
# 后缀
model2 = torch.load('./model/codeforces 936_16/TCN/embdDim3_lossL1Loss_optimAdam_hiddenDim5_startPos1_trainTypemix_nLayer1_dropout1/2024-09-24(17-43-45).pth')

data = InputData('data/codeforces 936_15.csv', 3, 'minute')#TODO 参数传递csv文件路径

vector_address='./vector/codeforces 936_16_vectors_2CBoW_noTime_noEnd_Vector_vLoss_v1.txt'
vocab_address='./vector/codeforces 936_15_2CBoW_noTime_noEnd_vocabulary.txt'
data.encodeEventByVocab(vocab_address, vector_address)

data.encodeTrace()
data.splitData(0.8)
data.initBatchData('minute', 1)

# 选择使用混合长度或者固定长度
# data.generateMixLengthBatch(batch_size=64)
data.generateSingleLengthBatch(batch_size=64, length_size=prefix_length)
data.initBatchData_tuijian('minute', 1)

ans = 0
self_input = []
self_input_road = []
# for i in range(len(data.input_roadset[2])):
'''
for i in range(64):
    # data.input_roadset 第一个表示前缀长度 第二个表示第几个轨迹 轨迹中0是前缀 1是后缀 2是target
    # print(data.input_roadset[2][i][0],data.input_roadset[2][i][1],data.input_roadset[2][i][2])
    self_input.append([data.event2id[data.input_roadset[2][i][0][0]], data.event2id[data.input_roadset[2][i][0][1]]])
    # self_input_road.append([data.event2id[data.input_roadset[2][i][1][0]], data.event2id[data.input_roadset[2][i][0][1]]])

    # 计算命中
    input_road = data.input_roadset[2][i][0]
    result = []
    for r in range(0, len(topic_order) + 1):
        for combo in combinations(topic_order, r):
            if not any(c in input_road for c in combo):
                result.append(list(combo))
    if data.input_roadset[2][i][1] in result:
        ans = ans + 1
self_input = np.array(self_input)
self_input = torch.LongTensor(self_input)
self_prediction = model(self_input)
for i in range(64):
    print(data.input_roadset[2][i][0], data.input_roadset[2][i][1], data.input_roadset[2][i][2], self_prediction[i].item())

print(ans,len(data.input_roadset[2]))

dif_road = list()
for i in range(len(data.input_roadset[2])):
    if data.input_roadset[2][i][1] not in dif_road:
        dif_road.append(data.input_roadset[2][i][1])
print(dif_road)

# detect(model, data.test_batch)
'''

self_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
self_dict_back = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
self_dict_score = {'A': 30, 'B': 30, 'C': 50, 'D': 50, 'E': 80, 'F': 80}
# self_dict = {'greedy implementation sortings': 1, 'greedy DP math': 2, 'greedy implementation DP binary_search trees': 3, 'greedy implementation bitmasks brute_force constructive_algorithms': 4, 'DP math number_theory combinatorics': 5, 'DP DFS data_structures 2-set': 6}
# self_dict_back = {1: 'greedy implementation sortings', 2: 'greedy DP math', 3: 'greedy implementation DP binary_search trees', 4: 'greedy implementation bitmasks brute_force constructive_algorithms', 5: 'DP math number_theory combinatorics', 6: 'DP DFS data_structures 2-set'}
# self_dict_score = {'greedy implementation sortings': 30, 'greedy DP math': 30, 'greedy implementation DP binary_search trees': 50, 'greedy implementation bitmasks brute_force constructive_algorithms': 50, 'DP math number_theory combinatorics': 80, 'DP DFS data_structures 2-set': 80}
# 打开文件，准备写入
with open("test/student_and_recommended_paths.txt", "w", encoding="utf-8") as f:
    sum_time = 0
    sum_score = 0
    num = 0
    score_sum_old = 0
    score_sum_new = 0
    point_old = 0
    point_new = 0
    sum_match = 0  # 记录匹配的节点数量
    total_nodes = 0  # 记录学生原始路径的节点总数
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(data.input_roadset[prefix_length])):
        # print("len(data.input_roadset[prefix_length])")
        # print(len(data.input_roadset[prefix_length]))
        if data.input_roadset[prefix_length][i][1] == []:
            continue;
        else:
            num = num + 1
        # self_input = []
        self_input = [[]]
        self_road = []
        result = []
        # 将 prefix_length 个元素添加到 self_input[0] 和 self_road
        for j in range(prefix_length):
            if j < len(data.input_roadset[prefix_length][i][0]):
                # 提取事件 ID 和节点值
                event_id = data.event2id[data.input_roadset[prefix_length][i][0][j][0]]
                node_value = data.input_roadset[prefix_length][i][0][j][0]

                # 将事件 ID 添加到 self_input 的内层列表中
                self_input[0].append(event_id)
                # 将节点值添加到 self_road
                self_road.append(node_value)
        # self_input.append([data.event2id[data.input_roadset[prefix_length][i][0][0]],
        #                    data.event2id[data.input_roadset[prefix_length][i][0][1]],
        #                    data.event2id[data.input_roadset[prefix_length][i][0][2]],
        #                    # data.event2id[data.input_roadset[prefix_length][i][0][3]],
        #                    # data.event2id[data.input_roadset[prefix_length][i][0][4]],
        #                    ])
        # print("self_input")
        # print(self_input)
        a_predict = getpredict(self_input, 1)
        # f.write(f"预测时间: {a_predict}\n")
        # self_road = [data.input_roadset[prefix_length][i][0][0][0],
        #              data.input_roadset[prefix_length][i][0][1][0],
        #              data.input_roadset[prefix_length][i][0][2][0],
        #              # data.input_roadset[prefix_length][i][0][3][0],
        #              # data.input_roadset[prefix_length][i][0][4][0],
        #              ]
        point_old = 0
        f.write(f"学生原始路径: {data.input_roadset[prefix_length][i][1]}\n")
        print("data.input_roadset[prefix_length][i][1]")
        print(data.input_roadset[prefix_length][i][1])

        for problem in data.input_roadset[prefix_length][i][1]:
            point_old = point_old + self_dict_score[problem]
        # print("score:",point_old - data.input_roadset[2][i][2])
        # 获取学习路径
        for r in range(1, len(topic_order) + 1):
            for combo in combinations(topic_order, r):
                if not any(c in self_road for c in combo):
                    result.append(list(combo))
        # 计算是否符合推荐条件
        c_point_new = 0
        point_new = 0
        best_recommended_path = []
        for road in result:
            # print("road")
            # print(road)
            max_point_new = 0
            self_input_road = []
            for j in range(len(road)):
                road[j] = self_dict[road[j]]
            self_input_road.append(road)
            road.append(6)
            road.append(6)
            # print("self_input_road")
            # print(self_input_road)
            b_predict = getpredict(self_input_road, 2)
            f.write(f"b_predict: {b_predict}\n")
            # print("b_predict")
            # print(b_predict)
            for j in range(len(road)):
                road[j] = self_dict_back[road[j]]
            f.write(f"data.input_roadset[prefix_length][i][2] + a_predict: {data.input_roadset[prefix_length][i][2] + a_predict}\n")
            if b_predict < data.input_roadset[prefix_length][i][2] + a_predict:
                if self_input_road[0][:-2] == data.input_roadset[prefix_length][i][1]:
                    sum_time = sum_time + 1
                c_point_new = 0
                self_input_road[0] = self_input_road[0][:-2]
                f.write(f"推荐路径T: {self_input_road[0]}\n")
                # print("self_input_road[0]")
                # print(self_input_road[0])
                for problem in self_input_road[0]:
                    c_point_new = c_point_new + self_dict_score[problem]
                point_new = max(c_point_new - b_predict, point_new)
                if point_new > max_point_new:
                    best_recommended_path = self_input_road[0]

        if best_recommended_path == data.input_roadset[prefix_length][i][1]:
            sum_score = sum_score + 1

        score_sum_old = score_sum_old + point_old
        score_sum_new = score_sum_new + point_new

        f.write(f"推荐路径T&S: {best_recommended_path}\n\n")

        # 获取两条路径的长度
        len_student_path = len(data.input_roadset[prefix_length][i][1])
        len_recommended_path = len(best_recommended_path)
        # if len_recommended_path < len_student_path:
        #     f.write(f"True");
        # 计算两条路径的最大长度
        max_len = max(len_student_path, len_recommended_path)
        # print("max_len", max_len)
        total_nodes += max_len
        total_nodes += prefix_length  # 全路径比较
        # 补齐学生路径
        padded_student_path = data.input_roadset[prefix_length][i][1] + [0] * (max_len - len_student_path)
        # 补齐推荐路径
        padded_recommended_path = best_recommended_path + [0] * (max_len - len_recommended_path)
        # # 使用 for 循环逐个位置比较路径
        # paths_are_equal = True
        # for j in range(max_len):
        #     if padded_recommended_path[j] != padded_student_path[j]:
        #         paths_are_equal = False
        #         break  # 如果有不匹配的节点，立即跳出循环
        # if paths_are_equal:
        #     sum_total += 1  # 如果路径完全一致，sum_total 加 1
        # 逐个位置进行比较，统计匹配的节点数量
        matches = sum([1 for j in range(max_len) if padded_recommended_path[j] == padded_student_path[j]])
        # print("matches", matches)
        sum_match += matches
        sum_match += prefix_length  # 全路径比较

        # 逐个位置比较并统计 TP, FP, FN
        for j in range(max_len):
            if padded_recommended_path[j] != padded_student_path[j]:
                if padded_recommended_path[j] != 0 and padded_student_path[j] != 0:
                    FP += 1  # 不一致且两个位置均不为0
                elif padded_recommended_path[j] == 0 or padded_student_path[j] == 0:
                    FN += 1  # 不一致且有一个位置为0

        # # 补齐推荐路径和学生原始路径
        # if best_recommended_path:
        #     # 去掉推荐路径末尾的两个节点
        #     # best_recommended_path = best_recommended_path[:-2]
        #
        #     # 获取两条路径的长度
        #     len_student_path = len(data.input_roadset[prefix_length][i][1])
        #     len_recommended_path = len(best_recommended_path)
        #
        #     # 计算两条路径的最大长度
        #     max_len = max(len_student_path, len_recommended_path)
        #
        #     # print("max_len", max_len)
        #
        #     total_nodes += max_len
        #     total_nodes += prefix_length  # 全路径比较
        #
        #     # 补齐学生路径
        #     padded_student_path = data.input_roadset[prefix_length][i][1] + [0] * (max_len - len_student_path)
        #     # 补齐推荐路径
        #     padded_recommended_path = best_recommended_path + [0] * (max_len - len_recommended_path)
        #
        #     # # 使用 for 循环逐个位置比较路径
        #     # paths_are_equal = True
        #     # for j in range(max_len):
        #     #     if padded_recommended_path[j] != padded_student_path[j]:
        #     #         paths_are_equal = False
        #     #         break  # 如果有不匹配的节点，立即跳出循环
        #     # if paths_are_equal:
        #     #     sum_total += 1  # 如果路径完全一致，sum_total 加 1
        #
        #     # 逐个位置进行比较，统计匹配的节点数量
        #     matches = sum([1 for j in range(max_len) if padded_recommended_path[j] == padded_student_path[j]])
        #     # print("matches", matches)
        #     sum_match += matches
        #     sum_match += prefix_length  # 全路径比较

TP = sum_match
# 计算 precision、recall、f1_score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("前缀长度：", prefix_length)
print("Student Number：", num)
print()
print("score_sum_old：", score_sum_old)
print("score_sum_new：", score_sum_new)
print("提升效果:", (score_sum_new - score_sum_old) / score_sum_old)
print("Accuracy_score：", sum_score / num)
print("Accuracy_time：", sum_time / num)
accuracy = sum_match / total_nodes if total_nodes > 0 else 0
print("准确率: {:.4%}".format(accuracy))
print("Precision: {:.4%}".format(precision))
print("Recall: {:.4%}".format(recall))
print("F1 Score: {:.4%}".format(f1_score))


