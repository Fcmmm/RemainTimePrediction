# -*- coding = UTF-8 -*-
# @Time : 2025/3/12 14:44
# @Author : 付传萌
# @File : trainVector_skipgram.py
# @Software : PyCharm

import numpy as np
from keras.layers import Flatten

np.random.seed(13)
import csv
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
# from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import time
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from datetime import datetime
import matplotlib.pyplot as plt
from keras.utils import plot_model
#from keras.utils.visualize_util import plot
#import pydot
#from pylab import *
from keras.models import Sequential, Model
# from keras.layers.core import Dense
# from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.layers.normalization import BatchNormalization
import numpy as np

eventlog = 'Codeforces 936_15'
dim = 3
window_size = 1

def readcsv(eventlog):
    csvfile = open('data/%s' % eventlog, 'r',encoding='utf-8')
    # print(csvfile)
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    sequence = []
    next(spamreader, None)  # skip the headers
    for line in spamreader:
        # 统一时间格式 23:03
        #line[2] = datetime.strptime(line[2], "%Y/%m/%d %H:%M").strftime("%Y-%m-%d %H:%M:%S")
        # print(line)
        sequence.append(line)
    # print(sequence)
    return sequence
data = readcsv(eventlog+'.csv')
# print(data)

def makeVocabulary(data, eventlog):
    temp = list()
    for line in data:
        temp.append(line[1])
    # ##print(temp)
    temp_temp = set(temp)
    # ##print(temp_temp)

    vocabulary = {sorted(list(temp_temp))[i]: i + 1 for i in range(len(temp_temp))}
    # vocabulary = {sorted(list(temp_temp))[i]:i for i in range(len(temp_temp))}
    # print(temp_temp,vocabulary)
    vocabulary['0'] = 0
    vocabulary['end'] = len(vocabulary)
    f = open('vector/%s' % eventlog + '_2skipgram_noTime_noEnd_vocabulary' + '.txt', 'w', encoding='utf-8')
    for k in vocabulary:
        f.write(str(k) + '\t' + str(vocabulary[k]) + '\n')
    return vocabulary


vocabulary = makeVocabulary(data, eventlog)
# ##print(v)

# 数据预处理
def processData(data,vocabulary,eventlog):
    front = data[0]
    data_new = []
    time_code_temp = {}
    time_code = {}
    for line in data[1:]:
        temp = 0
        #vocabulary_temp.append(line[1])
        if line[0] == front[0]:
            temp1 = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
            temp2 = time.strptime(front[2], "%Y-%m-%d %H:%M:%S")
            temp = datetime.fromtimestamp(time.mktime(temp1))-datetime.fromtimestamp(time.mktime(temp2))
        else:
            temp = 0
        t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
        week = datetime.fromtimestamp(time.mktime(t)).weekday()
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = datetime.fromtimestamp(time.mktime(t))-midnight
        data_new.append([line[0],vocabulary[str(line[1])],line[2],timesincemidnight,week])
        front = line
    front = data_new[0]
    for row in range(1, len(data_new)):
        line = data_new[row]
        if line[0] == front[0]:#id相同就进行以下操作（当前行数据-前一行数据）
            key = str(line[1]) + '-' + str(front[1])
            if key not in time_code_temp:
                time_code_temp[key] = []
                time_code_temp[key].append(line[3].seconds)
            else:
                time_code_temp[key].append(line[3].seconds)
        front = data_new[row]
    for key in time_code_temp:
        time_code_temp[key] =  sorted(time_code_temp[key])
    data_merge = []
    data_temp = [data_new[0]]
    for line in data_new[1:]:# 如果id不一样，就将该id的数据加入data_merge中，否则继续将该id的数据补充，直到下一个id
        if line[0] != data_temp[-1][0]:
            data_merge.append(data_temp)
            data_temp = [line]
        else:
            data_temp.append(line)
    data_merge.append(data_temp)

    vocabulary_num = len(vocabulary)

    vocabulary_temp = vocabulary
    return data_merge,data_new,time_code_temp,vocabulary_num,vocabulary_temp
data_merge,data_new,time_code_temp,vocabulary_num,vocabulary = processData(data,vocabulary,eventlog)
##print(data_new)
#     data_merge 每一行：每个用户的做题记录[用户ID, 事件编码, 时间戳, 距午夜秒数, 星期几]，[用户ID, 事件编码, 时间戳, 距午夜秒数, 星期几],……
#     data_new ：每个事件  [用户ID，事件编码，时间戳，距离午夜秒数，星期几]
#     time_code_temp：当前事件编码-前一个事件编码:[当前事件距离午夜秒数,···]
# 采样分 种
# 1.活动1，时间1，活动2，活动3，时间3
# 2.活动1，时间1，时间2，活动3，时间3
# 3.活动1，活动2，活动3，活动4，活动5


# 生成训练数据
def generate_data_skipgram(corpus, window_size, V):
    for words in corpus:
        L = len(words)
        for index, word in enumerate(words):
            contexts = []
            s = max(index - window_size, 0)
            e = min(index + window_size + 1, L)
            contexts = [words[i] for i in range(s, e) if i != index]
            center_word = word[1]  # 目标词的编码
            for context in contexts:
                context_word = context[1]
                x_activity = np.array([center_word])  # 目标词
                y_activity = np.eye(V)[context_word]  # 上下文词的独热编码
                yield (x_activity, y_activity)

X = []
Y = []
for x_activity, y_activity in generate_data_skipgram(data_merge, window_size, vocabulary_num):
    X.append(x_activity.tolist())
    Y.append(y_activity.tolist())

X = np.array(X)
Y = np.array(Y)

if __name__ == '__main__':
    skipgram = Sequential()
    skipgram.add(Embedding(input_dim=vocabulary_num, output_dim=dim, input_length=1))  # 输入长度改为1
    skipgram.add(Flatten())
    skipgram.add(Dense(vocabulary_num, activation='softmax'))  # 直接映射到词汇表
    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)
    skipgram.compile(loss='categorical_crossentropy', optimizer=opt)

    skipgram.fit(X, Y, validation_split=0.2, verbose=2,
                 batch_size=1, epochs=100)

    # 保存词向量
    f = open('vector/%s' % eventlog + '_vectors_SkipGram.txt', 'w')
    f.write('{} {}\n'.format(vocabulary_num, dim))
    vectors = skipgram.get_weights()[0]
    for word in range(vocabulary_num):
        str_vec = ' '.join(map(str, list(vectors[int(word), :])))

        f.write('{} {}\n'.format(word, str_vec))
    f.close()





