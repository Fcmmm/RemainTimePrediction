from tkinter import Variable
import numpy as np
import torch
import networkx as nx
from itertools import combinations, chain
from input_data import InputData
import matplotlib.pyplot as plt


def validate_data():
    total = len(data.input_roadset[prefix_length])
    valid = sum(1 for i in range(total) if data.input_roadset[prefix_length][i][1])
    print(f"总数据量: {total}")
    print(f"有效数据量: {valid}")
    print(f"数据有效率: {valid/total:.2%}")

def getpredict(list,m):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(64):
        if (len(list) < 64):
            list.append(list[0])
        else:
            break
    list = np.array(list)
    list = torch.LongTensor(list).to(device)

    if m == 1:
        prediction = model(list)
    else:
        prediction = model2(list)
    # print("prediction[0].item()",prediction[0].item())
    return prediction[0].item()

def detect(model, input_batchs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict_list = list()
    target_list = list()
    input_list = list()
    for (input,target) in input_batchs:
        # for i in range(len(input)):
        #    print(data.id2event[input[i][0]],data.id2event[input[i][1]])
        input = np.array(input)
        input = torch.LongTensor(input).to(device)

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
topics = ["A", "B", "C", "D", "E"]
# topics = ["greedy implementation sortings", "greedy DP math", "greedy implementation DP binary_search trees", "greedy implementation bitmasks brute_force constructive_algorithms", "DP math number_theory combinatorics", "DP DFS data_structures 2-set"]
G.add_nodes_from(topics, type="topic")

# 添加知识点节点
knowledge_points = ["math","bitmasks","constructive algorithms","greedy","brute force",
                    "data structures","dp","number theory","sortings","hashing",
                    "graphs","interactive"]
G.add_nodes_from(knowledge_points, type="知识点")

# 添加与知识点之间的连接关系
relations = {
    "A": ["math"],
    "B": ["bitmasks","constructive algorithms","greedy","math"],
    "C": ["brute force","data structures","dp","greedy","math","number theory","sortings"],
    "D": ["bitmasks","brute force","greedy","hashing"],
    "E": ["constructive algorithms","graphs","interactive"]
}


for topic, related_knowledge_points in relations.items():
    for kp in related_knowledge_points:
        G.add_edge(topic, kp)


'''# 创建图形
    plt.figure(figsize=(15, 10))

    # 设置布局
    pos = nx.spring_layout(G, k=1, iterations=50)

    # 绘制节点
    # 主题节点 - 红色
    topic_nodes = [node for node in G.nodes() if node in topics]
    nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes, node_color='red',
                          node_size=2000, alpha=0.7, node_shape='o')

    # 知识点节点 - 蓝色
    knowledge_nodes = [node for node in G.nodes() if node in knowledge_points]
    nx.draw_networkx_nodes(G, pos, nodelist=knowledge_nodes, node_color='lightblue',
                          node_size=1500, alpha=0.7, node_shape='s')

    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                          arrowsize=20, width=1, alpha=0.6)

    # 添加标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # 添加图例
    topic_patch = plt.Circle((0, 0), 0.02, color='red', alpha=0.7, label='Topics')
    knowledge_patch = plt.Rectangle((0, 0), 0.02, 0.02, color='lightblue', alpha=0.7, label='Knowledge Points')
    plt.legend(handles=[topic_patch, knowledge_patch], loc='upper left', bbox_to_anchor=(1, 1))

    # 设置标题和布局
    plt.title("Knowledge Graph Visualization", fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()

    # 显示图形
    plt.show()'''
# 使用拓扑排序来确定题目的顺序
topic_order = list(nx.topological_sort(G))[0:5]
print(topic_order)
# 前缀长度
prefix_length = 2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 前缀
model = torch.load('./model/codeforces 948_4/LSTMAtt/embdDim3_lossL1Loss_optimAdam_hiddenDim5_startPos1_trainTypeiteration_nLayer1_dropout1/2024-11-09(15-11-21).pth', map_location=device)#TODO 参数传递pth文件路径
# 后缀
model2 = torch.load('./model/codeforces 948_4/TCN/embdDim3_lossL1Loss_optimAdam_hiddenDim5_startPos1_trainTypemix_nLayer1_dropout1/2024-11-09(14-12-43).pth', map_location=device)

data = InputData('data/codeforces 948_4.csv', 3, 'minute') # TODO 参数传递csv文件路径

vector_address='./vector/codeforces 948_5_vectors_2CBoW_noTime_noEnd_Vector_vLoss_v1.txt'
vocab_address='./vector/codeforces 948_4_2CBoW_noTime_noEnd_vocabulary.txt'
data.encodeEventByVocab(vocab_address, vector_address)

data.encodeTrace()
data.splitData(0.7)
data.initBatchData('minute', 1)

# 选择使用混合长度或者固定长度
# data.generateMixLengthBatch(batch_size=64)
data.generateSingleLengthBatch(batch_size=64, length_size=prefix_length)
data.initBatchData_tuijian('minute', 1)

ans = 0
self_input = []
self_input_road = []


self_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
self_dict_back = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
self_dict_score = {'A': 20, 'B': 20, 'C': 50, 'D': 50, 'E': 80}
# self_dict = {'greedy implementation sortings': 1, 'greedy DP math': 2, 'greedy implementation DP binary_search trees': 3, 'greedy implementation bitmasks brute_force constructive_algorithms': 4, 'DP math number_theory combinatorics': 5, 'DP DFS data_structures 2-set': 6}
# self_dict_back = {1: 'greedy implementation sortings', 2: 'greedy DP math', 3: 'greedy implementation DP binary_search trees', 4: 'greedy implementation bitmasks brute_force constructive_algorithms', 5: 'DP math number_theory combinatorics', 6: 'DP DFS data_structures 2-set'}
# self_dict_score = {'greedy implementation sortings': 30, 'greedy DP math': 30, 'greedy implementation DP binary_search trees': 50, 'greedy implementation bitmasks brute_force constructive_algorithms': 50, 'DP math number_theory combinatorics': 80, 'DP DFS data_structures 2-set': 80}
# 打开文件，准备写入
with open("test/student_and_recommended_paths.txt", "w", encoding="utf-8") as f:
    sum_time = 0
    sum_score = 0
    num = 0
    num_recommended = 0  # 成功推荐的样本数
    score_sum_old = 0
    score_sum_new = 0
    point_old = 0
    point_new = 0
    sum_match = 0  # 记录匹配的节点数量
    total_nodes = 0  # 记录学生原始路径的节点总数
    TP = 0
    FP = 0
    FN = 0
    has_recommended_stu_num = 0
    print(len(data.input_roadset[prefix_length]))
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
        a_predict = getpredict(self_input, 1)
        f.write(f"预测时间: {a_predict}\n")
        # self_road = [data.input_roadset[prefix_length][i][0][0][0],
        #              data.input_roadset[prefix_length][i][0][1][0],
        #              data.input_roadset[prefix_length][i][0][2][0],
        #              # data.input_roadset[prefix_length][i][0][3][0],
        #              # data.input_roadset[prefix_length][i][0][4][0],
        #              ]
        point_old = 0
        f.write(f"学生原始路径: {data.input_roadset[prefix_length][i][1]}\n")

        # 计算学生原始路径分数
        student_score = 0
        for problem in data.input_roadset[prefix_length][i][1]:
            student_score = student_score + self_dict_score[problem]
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
        has_recommendation = False
        # print(result)
        for road in result:
            self_input_road = []
            for j in range(len(road)):
                road[j] = self_dict[road[j]]
            self_input_road.append(road)
            road.append(5)
            road.append(5)

            b_predict = getpredict(self_input_road, 2)
            f.write(f"b_predict: {b_predict}\n")
            for j in range(len(road)):
                road[j] = self_dict_back[road[j]]

            f.write(
                f"data.input_roadset[prefix_length][{i}][2] + a_predict: {data.input_roadset[prefix_length][i][2] + a_predict}\n")
            # 推荐路径的预测完成时间 < 原始路径的实际完成时间 + 前缀部分所预测的时间
            if b_predict < data.input_roadset[prefix_length][i][2] + a_predict:
                has_recommendation = True  # 标记找到了推荐路径
                if self_input_road[0][:-2] == data.input_roadset[prefix_length][i][1]:
                    sum_time = sum_time + 1
                c_point_new = 0
                self_input_road[0] = self_input_road[0][:-2]
                f.write(f"推荐路径T: {self_input_road[0]}\n")

                # 计算推荐路径分数
                for problem in self_input_road[0]:
                    c_point_new = c_point_new + self_dict_score[problem]
                point_new = max(c_point_new - b_predict, point_new)

                if point_new > 0:  # 如果找到更好的推荐路径
                    best_recommended_path = self_input_road[0]

            # 只在成功推荐路径的情况下累加分数
        if has_recommendation and best_recommended_path:
            has_recommended_stu_num += 1
            num_recommended += 1  # 记录成功推荐的数量
            score_sum_old += student_score  # 只累加有推荐的情况下的原始分数
            score_sum_new += point_new  # 累加推荐路径的分数
            if best_recommended_path == data.input_roadset[prefix_length][i][1]:
                sum_score += 1  # 只在有推荐且匹配时增加sum_score

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
        # print(total_nodes)
        # 补齐学生路径
        padded_student_path = data.input_roadset[prefix_length][i][1] + [0] * (max_len - len_student_path)
        # print(padded_student_path)
        # 补齐推荐路径
        padded_recommended_path = best_recommended_path + [0] * (max_len - len_recommended_path)
        # print(padded_recommended_path)
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

        # 补齐推荐路径和学生原始路径
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
print("recommended Number:",has_recommended_stu_num)
print("Find Rate:",has_recommended_stu_num/num)
print("score_sum_old：", score_sum_old)
print("score_sum_new：", score_sum_new)
print("提升效果:", (score_sum_new - score_sum_old) / score_sum_old if score_sum_old > 0 else 0)
print("Accuracy_score：",sum_score / num_recommended if num_recommended > 0 else 0)
print("Accuracy_time：", sum_time / num if num>0 else 0)
accuracy = sum_match / total_nodes if total_nodes > 0 else 0
print("准确率: {:.4%}".format(accuracy))
print("Precision: {:.4%}".format(precision))
print("Recall: {:.4%}".format(recall))
print("F1 Score: {:.4%}".format(f1_score))

# validate_data()
