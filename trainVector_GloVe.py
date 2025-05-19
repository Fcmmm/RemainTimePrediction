import numpy as np
import time
import warnings
import gensim
from datetime import datetime
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Dense, Lambda, Reshape, Dot, Add
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing import sequence
from keras.utils import plot_model
import csv

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# Event log
eventlog = 'Codeforces 936_15'
dim = 3
window_size = 1

def readcsv(eventlog):
    csvfile = open('data/%s' % eventlog, 'r', encoding='utf-8')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    sequence = []
    next(spamreader, None)  # skip the headers
    for line in spamreader:
        sequence.append(line)
    return sequence

data = readcsv(eventlog+'.csv')

# Vocabulary creation
def makeVocabulary(data, eventlog):
    temp = list()
    for line in data:
        temp.append(line[1])
    temp_temp = set(temp)
    vocabulary = {sorted(list(temp_temp))[i]: i + 1 for i in range(len(temp_temp))}
    vocabulary['0'] = 0
    vocabulary['end'] = len(vocabulary)
    f = open('vector/%s' % eventlog + 'glove_vocabulary' + '.txt', 'w', encoding='utf-8')
    for k in vocabulary:
        f.write(str(k) + '\t' + str(vocabulary[k]) + '\n')
    return vocabulary

vocabulary = makeVocabulary(data, eventlog)

# Data preprocessing
def processData(data, vocabulary, eventlog):
    front = data[0]
    data_new = []
    time_code_temp = {}
    time_code = {}
    for line in data[1:]:
        temp = 0
        if line[0] == front[0]:
            temp1 = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
            temp2 = time.strptime(front[2], "%Y-%m-%d %H:%M:%S")
            temp = datetime.fromtimestamp(time.mktime(temp1)) - datetime.fromtimestamp(time.mktime(temp2))
        else:
            temp = 0
        t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
        week = datetime.fromtimestamp(time.mktime(t)).weekday()
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = datetime.fromtimestamp(time.mktime(t)) - midnight
        data_new.append([line[0], vocabulary[str(line[1])], line[2], timesincemidnight, week])
        front = line
    front = data_new[0]
    for row in range(1, len(data_new)):
        line = data_new[row]
        if line[0] == front[0]:
            key = str(line[1]) + '-' + str(front[1])
            if key not in time_code_temp:
                time_code_temp[key] = []
                time_code_temp[key].append(line[3].seconds)
            else:
                time_code_temp[key].append(line[3].seconds)
        front = data_new[row]
    for key in time_code_temp:
        time_code_temp[key] = sorted(time_code_temp[key])
    data_merge = []
    data_temp = [data_new[0]]
    for line in data_new[1:]:
        if line[0] != data_temp[-1][0]:
            data_merge.append(data_temp)
            data_temp = [line]
        else:
            data_temp.append(line)
    data_merge.append(data_temp)
    vocabulary_num = len(vocabulary)
    vocabulary_temp = vocabulary
    return data_merge, data_new, time_code_temp, vocabulary_num, vocabulary_temp

data_merge, data_new, time_code_temp, vocabulary_num, vocabulary = processData(data, vocabulary, eventlog)

# Build co-occurrence matrix
def build_cooccurrence_matrix(corpus, vocab_size, window_size):
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for words in corpus:
        for i, center_word in enumerate(words):
            center_id = center_word[1]
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_id = words[j][1]
                    distance = abs(j - i)
                    cooccurrence_matrix[center_id, context_id] += 1.0 / distance
    return cooccurrence_matrix

# GloVe loss function
def glove_loss(cooccurrence):
    def loss(y_true, y_pred):
        weight = K.pow(K.clip(cooccurrence / 1, 0.0, 1.0), 0.75)
        return weight * K.square(y_pred - K.log(cooccurrence + 1))
    return loss

# Prepare training data
def prepare_training_data(cooccurrence_matrix):
    word_i_data = []
    word_j_data = []
    cooccurrence_data = []
    for i in range(len(cooccurrence_matrix)):
        for j in range(len(cooccurrence_matrix[i])):
            if cooccurrence_matrix[i, j] > 0:
                word_i_data.append(i)
                word_j_data.append(j)
                cooccurrence_data.append(cooccurrence_matrix[i, j])
    word_i_data = np.array(word_i_data)
    word_j_data = np.array(word_j_data)
    cooccurrence_data = np.array(cooccurrence_data)
    return word_i_data, word_j_data, cooccurrence_data

# Build GloVe model
def build_glove_model(vocab_size, dim):
    word_i = Input(shape=(1,), name='word_i')
    word_j = Input(shape=(1,), name='word_j')
    cooccurrence = Input(shape=(1,), name='cooccurrence')
    w_i = Embedding(vocab_size, dim, name='w_i')(word_i)
    w_j = Embedding(vocab_size, dim, name='w_j')(word_j)
    b_i = Embedding(vocab_size, 1, name='b_i')(word_i)
    b_j = Embedding(vocab_size, 1, name='b_j')(word_j)
    w_i = Reshape((dim,))(w_i)
    w_j = Reshape((dim,))(w_j)
    b_i = Reshape((1,))(b_i)
    b_j = Reshape((1,))(b_j)
    dot_product = Dot(axes=1)([w_i, w_j])
    predict = Add()([dot_product, b_i, b_j])
    model = Model(inputs=[word_i, word_j, cooccurrence], outputs=predict)
    return model

# Main training loop
if __name__ == '__main__':
    cooccurrence_matrix = build_cooccurrence_matrix(data_merge, vocabulary_num, window_size)
    word_i_data, word_j_data, cooccurrence_data = prepare_training_data(cooccurrence_matrix)
    glove_model = build_glove_model(vocabulary_num, dim)
    opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    glove_model.compile(loss=glove_loss(cooccurrence_data), optimizer=opt)

    early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)
    model_checkpoint = ModelCheckpoint('vector/%s' % eventlog + '_GloVe_Model_{epoch:02d}-{val_loss:.2f}.h5',
                                       monitor='val_loss', save_best_only=True, save_weights_only=False)

    glove_model.fit([word_i_data, word_j_data, cooccurrence_data],
                    cooccurrence_data,
                    validation_split=0.2,
                    epochs=500,
                    batch_size=512,
                    callbacks=[early_stopping, model_checkpoint, lr_reducer])

    word_vectors = glove_model.get_layer('w_i').get_weights()[0] + glove_model.get_layer('w_j').get_weights()[0]
    with open('vector/%s' % eventlog + '_vectors_GloVe.txt', 'w') as f:
        f.write('{} {}\n'.format(vocabulary_num, dim))
        for word_id in range(vocabulary_num):
            vec = word_vectors[word_id]
            f.write('{} {}\n'.format(word_id, ' '.join(map(str, vec))))
