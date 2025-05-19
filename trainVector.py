import numpy as np
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
    f = open('vector/%s' % eventlog + '_2CBoW_noTime_noEnd_vocabulary' + '.txt', 'w', encoding='utf-8')
    for k in vocabulary:
        f.write(str(k) + '\t' + str(vocabulary[k]) + '\n')
    return vocabulary


vocabulary = makeVocabulary(data, eventlog)
# ##print(v)

# 学习事件数据预处理
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
def generate_data(corpus, window_size, V):
    maxlen = window_size * 2
    flag = 0
    for words in corpus:
        L = len(words)
        for index, word in enumerate(words):
            contexts = []
            labels = []
            s = index - window_size
            e = index + window_size + 1
            contexts = [words[i] for i in range(s, e) if 0 <= i < L and i != index]
            contexts_activity = [context[1] for context in contexts]
            labels.append(word)
            labels_activity = word[1]
            x_activity = sequence.pad_sequences([contexts_activity], maxlen=maxlen)
            y_activity = np.eye(V)[labels_activity]
            yield (x_activity, y_activity)

X = []
Y = []
for x_activity,y_activity in generate_data(data_merge, window_size, vocabulary_num):
    X.append(x_activity.reshape(-1).tolist())
    Y.append(y_activity.reshape(-1).tolist())
# 得到的X为上下文的标签，Y为对应的独热编码   X[1,3]  Y[0 0 1 0 0 0 0 0]
X = np.array(X)
Y = np.array(Y)
if __name__ == '__main__':

    cbow = Sequential()
    cbow.add(Embedding(input_dim=vocabulary_num, output_dim=dim, input_length=window_size*2))
    cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(dim,)))
    cbow.add(Dense(vocabulary_num, activation='softmax'))
    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)
    cbow.compile(loss='categorical_crossentropy', optimizer=opt)
    early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100,
                                   verbose=0, mode='auto',  cooldown=0, min_lr=0)
    model_checkpoint = ModelCheckpoint('vector/%s' % eventlog+'_2CBoW_noTime_noEnd_Vector_vLoss_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_accuracy',
                                       verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    cbow.fit(X, Y, validation_split=0.2, verbose=2,
              callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=1, epochs=100)
    f = open('vector/%s' % eventlog + '_vectors_2CBoW_noTime_noEnd_Vector_vLoss_v1' + '.txt', 'w')
    f.write('{} {}\n'.format(vocabulary_num, dim))
    vectors = cbow.get_weights()[0]
    for word in range(vocabulary_num):
        str_vec = ' '.join(map(str, list(vectors[int(word), :])))
        f.write('{} {}\n'.format(word, str_vec))
    f.close()






