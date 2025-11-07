import numpy as np
import torch
import torch.nn as nn
import gensim
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt
import random
random.seed=13
class InputData():
    def __init__(self,data_address, embd_dimension = 3, time_unit = 'second'):
        self.embedding = None
        self.ogrinal_data = list()
        self.orginal_trace = list()
        self.encode_trace = list()
        self.train_dataset = list()
        self.test_dataset = list()
        self.train_mixLengthData = list()
        self.test_mixLengthData = list()
        self.event2id = dict()
        self.id2event = dict()
        self.train_batch_mix = list()
        self.test_batch_mix = list()
        self.train_singleLengthData = dict()
        self.test_singleLengthData = dict()
        self.train_batch_num = 0
        # 已完成前缀
        self.input_roadset = list()
        # 实际路径
        self.test_roadset = list()

        self.train_batch = dict()
        self.test_batch = dict()
        self.train_batch_single = dict()
        self.test_batch_single = dict()

        self.vocab_size = 0
        self.train_maxLength = 0
        self.test_maxLength = 0
        self.embd_dimension = embd_dimension
        if time_unit == 'second':
            self.time_unit = 1
        elif time_unit == 'minute':
            self.time_unit = 60
        elif time_unit == 'hour':
            self.time_unit = 60 * 60
        elif time_unit == 'day':
            self.time_unit = 24 * 60 * 60
        elif time_unit == 'month':
            self.time_unit = 30 * 24 * 60 * 60
        self.initData(data_address)
    # 初始 ogrinal_data  orginal_trace
    def initData(self,data_address):
        # 初始化数组
        orginal_trace = list() # 结果数组
        record = list() # 原始数据数组
        trace_temp = list() # 过程数组
        # 读入原始数据 放入record中
        with open(data_address, 'r', encoding='utf-8') as f:
            next(f)
            lines = f.readlines()
            for line in lines:
                record.append(line)
        # 获取第一个用户的id
        flag = record[0].split(',')[0]
        for line in record:
            line = line.replace('\r', '').replace('\n', '')
            line = line.split(',')
            # 如果这行记录输入该用户则加入过程数组
            if line[0] == flag:
                # 加入该行的信息：学习知识点、时间戳
                trace_temp.append([line[1], line[2]])
            else:
            # 否则说明当前flag所记录的用户所有提交记录已经进入过程数组 则更新flag为下一个用户 将过程数组复制到结果数组
                flag = line[0]
                if len(trace_temp) > 0:
                    orginal_trace.append(trace_temp.copy())
                trace_temp = list()
                trace_temp.append([line[1], line[2]])
        self.ogrinal_data = record
        self.orginal_trace = orginal_trace
        # print(len(ogrinal_data))
    def encodeEvent(self,vector_address):
        # 定义转换的字典
        event2id = dict()
        id2event = dict()
        if vector_address == None:
            for line in self.ogrinal_data:
                line = line.replace('\r', '').replace('\n', '')
                line = line.split(',')
                try:
                    event2id[line[1]] = event2id[line[1]]
                    id2event[event2id[line[1]]] = id2event[event2id[line[1]]]
                except KeyError as ke:
                    # 将所学知识点转化成id 转化格式为event2id的当前长度
                    event2id[line[1]] = len(event2id)
                    id2event[len(id2event)] = line[1]
            self.vocab_size = len(event2id)
            self.embedding = nn.Embedding(self.vocab_size + 1, self.embd_dimension, padding_idx= self.vocab_size)
            # print(event2id,id2event,self.embedding,self.vocab_size)
        else:
            with open(vector_address, 'r', encoding='utf-8') as f:
                information = next(f)
                information = information.replace('\n','').replace('\r','').split(' ')
                self.vocab_size = int(information[0])
                self.embd_dimension = int(information[1])
                lines = f.readlines()
                for line in lines:
                    line = line.split(' ')
                    event2id[line[0]] = len(event2id)
                    id2event[len(id2event)] = line[0]
            # 加载预训练词向量
            wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
                vector_address, binary=False, encoding='utf-8')
            # print(len(wvmodel.index_to_key))
            weight = torch.zeros(self.vocab_size + 1, self.embd_dimension)
            for i in range(len(wvmodel.index_to_key)):
                weight[i, :] = torch.from_numpy(wvmodel.get_vector(
                    wvmodel.index_to_key[i]))
            self.embedding = nn.Embedding.from_pretrained(weight, padding_idx=self.vocab_size)
        self.event2id = event2id
        self.id2event = id2event

    def encodeEventByVocab(self, vocab_address, vector_address):
        v_o = open(vocab_address, 'r', encoding='utf-8')
        v_r = v_o.readlines()
        event2id = dict()
        id2event = dict()
        for line in v_r:
            line = line.replace('\r', '')
            line = line.replace('\n', '')
            line = line.split('\t')
            event2id[line[0]] = int(line[1])
            id2event[int(line[1])] = line[0]
        v_o.close()
        self.event2id = event2id
        self.id2event = id2event
        # 加载预训练词向量
        wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
            vector_address, binary=False, encoding='utf-8')
        # print(wvmodel.index_to_key)
        # num_classes = len(wvmodel.index_to_key)
        self.vocab_size = len(wvmodel.index_to_key)
        weight = torch.zeros(self.vocab_size + 1, self.embd_dimension)
        for i in range(len(wvmodel.index_to_key)):
            weight[i, :] = torch.from_numpy(wvmodel.get_vector(
                wvmodel.index_to_key[i]))
        self.embedding = nn.Embedding.from_pretrained(weight, padding_idx=self.vocab_size)
        # print(event2id, id2event, self.embedding, self.vocab_size,self.embedding.weight)
    def encodeTrace(self):
        # 将orginal_trace复制一份同时对所学知识点进行编码，记录序列最大长度
        encode_trace = list()
        max = 0
        for line in self.orginal_trace:
            trace_temp = list()
            for line2 in line:
                # print([self.event2id[line2[0]], line2[1]])
                trace_temp.append([self.event2id[line2[0]], line2[1]])
            if len(trace_temp) > max:
                max = len(trace_temp)
            encode_trace.append(trace_temp.copy())
        self.max = max
        self.encode_trace = encode_trace
        #[[[1, '2024-03-22 18:07:00'], [2, '2024-03-22 18:48:00']], [[1, '2024-03-22 17:45:00'], [2, '2024-03-22 18:10:00'], [3, '2024-03-22 19:10:00']],
    def splitData(self,train_splitThreshold = 1):
        self.train_dataset, self.test_dataset = train_test_split(self.encode_trace, train_size=train_splitThreshold, test_size=1-train_splitThreshold)
    def initBatchData(self, time_unit, start_pos):
        if time_unit == 'second':
            time_unit = 1
        elif time_unit == 'minute':
            time_unit = 60
        elif time_unit == 'hour':
            time_unit = 60 * 60
        elif time_unit == 'day':
            time_unit = 24 * 60 * 60
        elif time_unit == 'month':
            time_unit = 30 * 24 * 60 * 60
        # 单一长度训练、测试集
        train_singleLengthData = dict()
        test_singleLengthData = dict()
        # 混合长度训练、测试集
        train_mixLengthData = list()
        test_mixLengthData = list()
        # 最大长度
        train_maxLength = 0
        test_maxLength = 0


        # 前缀
        for line in self.train_dataset:
            train_input_temp = list()
            for line2 in line:
                train_input_temp.append(line2[0])
                if len(train_input_temp) > train_maxLength:
                    train_maxLength = len(train_input_temp)
                # 时间格式化 -> 剩余时间
                target_time = abs((datetime.strptime(str(line2[1]), '%Y/%m/%d %H:%M') - \
                               datetime.strptime(str(line[-1][1]), '%Y/%m/%d %H:%M')).total_seconds() / time_unit)
                if target_time >= 60 :
                    continue

                try:
                    train_singleLengthData[len(train_input_temp)].append((train_input_temp.copy(), target_time))
                except BaseException as e:
                    # 轨迹长度不存在时
                    train_singleLengthData[len(train_input_temp)] = list()
                    train_singleLengthData[len(train_input_temp)].append((train_input_temp.copy(), target_time))
                if len(train_input_temp) >= start_pos:
                    train_mixLengthData.append((train_input_temp.copy(), target_time))
                # print(train_input_temp.copy(),target_time)
        # print(train_mixLengthData)
        for line in self.test_dataset:
            # 测试集
            test_input_temp = list()
            for line2 in line:
                test_input_temp.append(line2[0])

                if len(test_input_temp) > test_maxLength:
                    test_maxLength = len(test_input_temp)
                target_time = abs((datetime.strptime(str(line2[1]), '%Y/%m/%d %H:%M') - \
                               datetime.strptime(str(line[-1][1]), '%Y/%m/%d %H:%M')).total_seconds() / time_unit)
                if target_time >=60:
                    continue

                try:
                    test_singleLengthData[len(test_input_temp)].append((test_input_temp.copy(), target_time))
                except BaseException as e:
                    test_singleLengthData[len(test_input_temp)] = list()
                    test_singleLengthData[len(test_input_temp)].append((test_input_temp.copy(), target_time))
                if len(test_input_temp) >= start_pos:
                    test_mixLengthData.append((test_input_temp.copy(), target_time))




        # 后缀
        '''for line in self.train_dataset:
            train_input_temp = list()
            line_len = len(line)
            for i in range(1,line_len + 1):
                # 逆序
                train_input_temp.append(line[line_len - i][0])
                if len(train_input_temp) > train_maxLength:
                    train_maxLength = len(train_input_temp)
                # 时间格式化
                time_be = datetime.strptime('2024-03-22 17:35:00', '%Y-%m-%d %H:%M:%S')
                if line_len - i - 1 >= 0 :
                    time_be = datetime.strptime(str(line[line_len - i - 1][1]), '%Y-%m-%d %H:%M:%S')
                    # print(line_len,line_len-i-1,time_be)
                time_en = datetime.strptime(str(line[-1][1]),'%Y-%m-%d %H:%M:%S')
                target_time = abs((time_be - time_en).total_seconds() / time_unit)
                add_temp = list()
                # TCN补长
                for step in range(len(train_input_temp), 3):
                    add_temp.append(train_input_temp[len(train_input_temp) - 1])
                try:
                    train_singleLengthData[len(train_input_temp)].append((train_input_temp[::-1].copy() + add_temp, target_time))
                except BaseException as e:
                    train_singleLengthData[len(train_input_temp)] = list()
                    train_singleLengthData[len(train_input_temp)].append((train_input_temp[::-1].copy() + add_temp, target_time))
                if len(train_input_temp) >= start_pos:
                    train_mixLengthData.append((train_input_temp[::-1].copy() + add_temp, target_time))
        for line in self.test_dataset:
            test_input_temp = list()
            line_len = len(line)
            # print(self.test_dataset)
            for i in range(1, line_len + 1):
                test_input_temp.append(line[line_len - i][0])
                # print(test_input_temp)
                if len(test_input_temp) > test_maxLength:
                    test_maxLength = len(test_input_temp)
                # 2024-05-26 18:12:00
                # 2024-03-22 17:35:00
                time_be = datetime.strptime('2024-03-22 17:35:00', '%Y-%m-%d %H:%M:%S')
                if line_len - i - 1 >= 0:
                    time_be = datetime.strptime(str(line[line_len - i - 1][1]), '%Y-%m-%d %H:%M:%S')
                time_en = datetime.strptime(str(line[-1][1]), '%Y-%m-%d %H:%M:%S')
                target_time = abs((time_be - time_en).total_seconds() / time_unit)
                add_temp = list()
                # TCN补长
                print(target_time)
                for step in range(len(train_input_temp), 3):
                    add_temp.append(train_input_temp[len(train_input_temp) - 1])
                try:
                    test_singleLengthData[len(test_input_temp)].append((test_input_temp[::-1].copy() + add_temp, target_time))
                except BaseException as e:
                    test_singleLengthData[len(test_input_temp)] = list()
                    test_singleLengthData[len(test_input_temp)].append((test_input_temp[::-1].copy() + add_temp, target_time))
                if len(test_input_temp) >= start_pos:
                    test_mixLengthData.append((test_input_temp[::-1].copy() + add_temp, target_time))'''
        # print(train_mixLengthData)
        self.train_singleLengthData = train_singleLengthData
        self.test_singleLengthData = test_singleLengthData
        self.train_mixLengthData = train_mixLengthData
        self.test_mixLengthData = test_mixLengthData
        self.train_maxLength = train_maxLength
        self.test_maxLength = test_maxLength
    def generateSingleLengthBatch(self,batch_size,length_size):
        train_batch_single = list()
        test_batch_single = list()
        input_temp = list()
        target_temp = list()
        max_length = 0
        if length_size in self.train_batch_single:
            self.train_batch = self.train_batch_single[length_size]
            self.test_batch = self.test_batch_single[length_size]
        for line in self.train_singleLengthData[length_size]:
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                train_batch_single.append((input_temp.copy(),target_temp.copy()))
                max_length = 0
                input_temp = list()
                target_temp = list()
                input_temp.append(line[0])
                target_temp.append(line[1])
            input_temp.append(line[0])
            target_temp.append(line[1])
        while len(input_temp) < batch_size:
            if len(train_batch_single) == 0 and len(input_temp) == 0:
                break
            elif len(train_batch_single) == 0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(input_temp) - 1)
                input_temp.append(input_temp[ran1])
                target_temp.append(target_temp[ran1])
            elif len(train_batch_single) !=0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(train_batch_single)-1)
                (ran_input,ran_target) = train_batch_single[ran1]
                ran2 = random.randint(0, len(train_batch_single[ran1])-1)
                input_temp.append(ran_input[ran2])
                target_temp.append(ran_target[ran2])
        train_batch_single.append((input_temp.copy(), target_temp.copy()))
        max_length = 0
        input_temp = list()
        target_temp = list()
        for line in self.test_singleLengthData[length_size]:
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                test_batch_single.append((input_temp.copy(),target_temp.copy()))
                max_length = 0
                input_temp = list()
                target_temp = list()
                input_temp.append(line[0])
                target_temp.append(line[1])
            input_temp.append(line[0])
            target_temp.append(line[1])
        while len(input_temp) < batch_size:
            #print(len(test_batch_single),test_batch_single)
            if len(test_batch_single) ==0 and len(input_temp) == 0:
                break
            elif len(test_batch_single) ==0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(input_temp) - 1)
                input_temp.append(input_temp[ran1])
                target_temp.append(target_temp[ran1])
            elif len(test_batch_single) !=0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(test_batch_single)-1)
                (ran_input,ran_target) = test_batch_single[ran1]
                ran2 = random.randint(0, len(test_batch_single[ran1])-1)
                input_temp.append(ran_input[ran2])
                target_temp.append(ran_target[ran2])
        test_batch_single.append((input_temp.copy(), target_temp.copy()))
        # print(train_batch_single)
        self.train_batch_single[length_size] = train_batch_single
        self.test_batch_single[length_size] = test_batch_single
        self.train_batch = self.train_batch_single[length_size]
        self.test_batch = self.test_batch_single[length_size]
    def generateMixLengthBatch(self, batch_size):
        train_batch = list()
        test_batch = list()
        input_temp = list()
        target_temp = list()
        max_length = 0
        if len(self.test_batch_mix) > 0:
            self.train_batch = self.train_batch_mix
            self.test_batch = self.test_batch_mix
            return 0
        for line in self.train_mixLengthData:
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                train_batch.append((input_temp.copy(),target_temp.copy()))
                max_length = 0
                input_temp = list()
                target_temp = list()
            input_temp.append(line[0])
            target_temp.append(line[1])
            if max_length < len(line[0]):
                max_length = len(line[0])
        while len(input_temp) < batch_size:
            ran1 = random.randint(0, len(train_batch)-1)
            (ran_input,ran_target) = train_batch[ran1]
            ran2 = random.randint(0, len(train_batch[ran1])-1)
            input_temp.append(ran_input[ran2].copy())
            target_temp.append(ran_target[ran2])
        max_length = 0
        for line in input_temp:
            if max_length < len(line):
                max_length = len(line)
        for num in range(len(input_temp)):
            while len(input_temp[num]) < max_length:
                input_temp[num].append(self.vocab_size)
        train_batch.append((input_temp.copy(), target_temp.copy()))
        max_length = 0
        input_temp = list()
        target_temp = list()
        for line in self.test_mixLengthData:
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                test_batch.append((input_temp.copy(),target_temp.copy()))
                max_length = 0
                input_temp = list()
                target_temp = list()
            input_temp.append(line[0])
            target_temp.append(line[1])
            if max_length < len(line[0]):
                max_length = len(line[0])

        while len(input_temp) < batch_size:
            ran1 = random.randint(0, len(test_batch)-1)
            (ran_input,ran_target) = test_batch[ran1]
            ran2 = random.randint(0, len(test_batch[ran1])-1)
            input_temp.append(ran_input[ran2].copy())
            target_temp.append(ran_target[ran2])
        max_length = 0
        for line in input_temp:
            if max_length < len(line):
                max_length = len(line)

        for num in range(len(input_temp)):
            while len(input_temp[num]) < max_length:
                input_temp[num].append(self.vocab_size)
        test_batch.append((input_temp.copy(), target_temp.copy()))
        # print(train_batch)
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.train_batch_mix = train_batch
        self.test_batch_mix = test_batch

# 资源推荐补充代码
    def initBatchData_tuijian(self, time_unit, start_pos):
        if time_unit == 'second':
            time_unit = 1
        elif time_unit == 'minute':
            time_unit = 60
        elif time_unit == 'hour':
            time_unit = 60 * 60
        elif time_unit == 'day':
            time_unit = 24 * 60 * 60
        elif time_unit == 'month':
            time_unit = 30 * 24 * 60 * 60
        # 单一长度前缀路径、实际后缀
        input_roadset = dict()
        # 最大长度
        input_road_maxLength = 0
        test_maxLength = 0

        for line in self.encode_trace:
            # 已完成路径
            # 单个轨迹 [[2, '2024-05-26 18:27:00'], [3, '2024-05-26 18:39:00']]
            input_road_temp = list()
            test_road_temp = list()
            for line2 in line:
                # line2 单个题目信息[2, '2024-05-26 18:27:00']
                input_road_temp.append(self.id2event[line2[0]])

                if len(input_road_temp) > input_road_maxLength:
                    input_road_maxLength = len(input_road_temp)
                # 时间格式化 -> 剩余时间
                target_time = abs((datetime.strptime(str(line2[1]), '%Y-%m-%d %H:%M:%S') - \
                               datetime.strptime(str(line[-1][1]), '%Y-%m-%d %H:%M:%S')).total_seconds() / time_unit)
                # if target_time == 0 :
                    # continue;
                # 获取实际后缀
                for bj in range(len(input_road_temp), len(line)):
                    test_road_temp.append(self.id2event[line[bj][0]][0])

                # 记录前缀、后缀和实际用时
                try:
                    input_roadset[len(input_road_temp)].append((input_road_temp.copy(), test_road_temp.copy(), target_time))
                except BaseException as e:
                    input_roadset[len(input_road_temp)] = list()
                    input_roadset[len(input_road_temp)].append((input_road_temp.copy(), test_road_temp.copy(), target_time))

                test_road_temp = list()
        # print(input_roadset)
        self.input_roadset = input_roadset

    def generateSingleLengthBatch_tuijian(self,batch_size,length_size):
        train_batch_single = list()
        test_batch_single = list()
        input_temp = list()
        target_temp = list()
        max_length = 0
        if length_size in self.train_batch_single:
            self.train_batch = self.train_batch_single[length_size]
            self.test_batch = self.test_batch_single[length_size]

        for line in self.train_singleLengthData[length_size]:
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                train_batch_single.append((input_temp.copy(),target_temp.copy()))
                max_length = 0
                input_temp = list()
                target_temp = list()
                input_temp.append(line[0])
                target_temp.append(line[1])
            input_temp.append(line[0])
            target_temp.append(line[1])
        while len(input_temp) < batch_size:
            if len(train_batch_single) == 0 and len(input_temp) == 0:
                break
            elif len(train_batch_single) == 0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(input_temp) - 1)
                input_temp.append(input_temp[ran1])
                target_temp.append(target_temp[ran1])
            elif len(train_batch_single) !=0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(train_batch_single)-1)
                (ran_input,ran_target) = train_batch_single[ran1]
                ran2 = random.randint(0, len(train_batch_single[ran1])-1)
                input_temp.append(ran_input[ran2])
                target_temp.append(ran_target[ran2])
        train_batch_single.append((input_temp.copy(), target_temp.copy()))
        max_length = 0
        input_temp = list()
        target_temp = list()
        for line in self.test_singleLengthData[length_size]:
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                test_batch_single.append((input_temp.copy(),target_temp.copy()))
                max_length = 0
                input_temp = list()
                target_temp = list()
                input_temp.append(line[0])
                target_temp.append(line[1])
            input_temp.append(line[0])
            target_temp.append(line[1])
        while len(input_temp) < batch_size:
            #print(len(test_batch_single),test_batch_single)
            if len(test_batch_single) ==0 and len(input_temp) == 0:
                break
            elif len(test_batch_single) ==0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(input_temp) - 1)
                input_temp.append(input_temp[ran1])
                target_temp.append(target_temp[ran1])
            elif len(test_batch_single) !=0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(test_batch_single)-1)
                (ran_input,ran_target) = test_batch_single[ran1]
                ran2 = random.randint(0, len(test_batch_single[ran1])-1)
                input_temp.append(ran_input[ran2])
                target_temp.append(ran_target[ran2])
        test_batch_single.append((input_temp.copy(), target_temp.copy()))
        #print(test_batch_single)
        self.train_batch_single[length_size] = train_batch_single
        self.test_batch_single[length_size] = test_batch_single
        self.train_batch = self.train_batch_single[length_size]
        self.test_batch = self.test_batch_single[length_size]




