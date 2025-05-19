import numpy as np

np.random.seed(13)
import csv
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda, Flatten, Dropout, Concatenate
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import time
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

eventlog = 'Codeforces 936_15'
dim = 3
window_size = 1
min_n = 3  # Minimum n-gram size
max_n = 6  # Maximum n-gram size


def readcsv(eventlog):
    csvfile = open('data/%s' % eventlog, 'r', encoding='utf-8')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    sequence = []
    next(spamreader, None)  # skip the headers
    for line in spamreader:
        sequence.append(line)
    return sequence


data = readcsv(eventlog + '.csv')


def makeVocabulary(data, eventlog):
    temp = list()
    for line in data:
        temp.append(line[1])
    temp_temp = set(temp)

    vocabulary = {sorted(list(temp_temp))[i]: i + 1 for i in range(len(temp_temp))}
    vocabulary['0'] = 0
    vocabulary['end'] = len(vocabulary)
    f = open('vector/%s' % eventlog + '_FastText_noTime_noEnd_vocabulary' + '.txt', 'w', encoding='utf-8')
    for k in vocabulary:
        f.write(str(k) + '\t' + str(vocabulary[k]) + '\n')
    return vocabulary


vocabulary = makeVocabulary(data, eventlog)

# Create a reverse vocabulary (id -> token)
rev_vocabulary = {v: k for k, v in vocabulary.items()}


# 数据预处理
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


# Generate character n-grams for FastText
def generate_char_ngrams(word, min_n=min_n, max_n=max_n):
    if not isinstance(word, str):
        # Convert to string if it's a number or other type
        word = str(word)

    char_ngrams = []
    for n in range(min_n, min(max_n + 1, len(word) + 1)):
        for i in range(len(word) - n + 1):
            char_ngrams.append(word[i:i + n])
    return char_ngrams


# Create character n-gram dictionary
char_ngrams_dict = {}
ngram_id = vocabulary_num
for word_id in range(vocabulary_num):
    if word_id in rev_vocabulary:
        word = rev_vocabulary[word_id]
        char_ngrams = generate_char_ngrams(word)
        for ngram in char_ngrams:
            if ngram not in char_ngrams_dict:
                char_ngrams_dict[ngram] = ngram_id
                ngram_id += 1

# Mapping from word_id to its character n-grams ids
word_to_ngrams = {}
for word_id in range(vocabulary_num):
    if word_id in rev_vocabulary:
        word = rev_vocabulary[word_id]
        char_ngrams = generate_char_ngrams(word)
        word_to_ngrams[word_id] = [char_ngrams_dict[ngram] for ngram in char_ngrams]

# Total vocabulary size including character n-grams
total_vocab_size = len(vocabulary) + len(char_ngrams_dict)


# 生成训练数据
def generate_data(corpus, window_size, V):
    maxlen = window_size * 2
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

            # Get character n-grams for the target word
            target_ngrams = word_to_ngrams.get(labels_activity, [])

            yield (x_activity, y_activity, labels_activity, target_ngrams)


# Collect data for training
X = []
Y = []
word_ids = []
ngram_ids = []

for x_activity, y_activity, word_id, target_ngrams in generate_data(data_merge, window_size, vocabulary_num):
    X.append(x_activity.reshape(-1).tolist())
    Y.append(y_activity.reshape(-1).tolist())
    word_ids.append(word_id)
    ngram_ids.append(target_ngrams)

X = np.array(X)
Y = np.array(Y)

if __name__ == '__main__':
    # FastText model
    # Input for context words
    context_input = Input(shape=(window_size * 2,), name='context_input')

    # Word embeddings
    word_embedding = Embedding(input_dim=total_vocab_size, output_dim=dim, name='word_embedding')(context_input)
    word_context = Lambda(lambda x: K.mean(x, axis=1), output_shape=(dim,))(word_embedding)

    # Model for predicting with context
    output = Dense(vocabulary_num, activation='softmax', name='output')(word_context)

    fasttext_model = Model(inputs=context_input, outputs=output)

    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)
    fasttext_model.compile(loss='categorical_crossentropy', optimizer=opt)

    early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100,
                                   verbose=0, mode='auto', cooldown=0, min_lr=0)
    model_checkpoint = ModelCheckpoint(
        'vector/%s' % eventlog + '_FastText_noTime_noEnd_Vector_vLoss_{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto')

    fasttext_model.fit(X, Y, validation_split=0.2, verbose=2,
                       callbacks=[early_stopping, model_checkpoint, lr_reducer],
                       batch_size=1, epochs=100)

    # Save word vectors
    f = open('vector/%s' % eventlog + '_vectors_FastText_noTime_noEnd_Vector_vLoss_v1' + '.txt', 'w')
    f.write('{} {}\n'.format(vocabulary_num, dim))

    word_vectors = fasttext_model.get_weights()[0]

    # Save both word vectors and subword n-gram vectors
    for word_id in range(vocabulary_num):
        word_vec = word_vectors[word_id]
        str_vec = ' '.join(map(str, list(word_vec)))
        f.write('{} {}\n'.format(word_id, str_vec))

    # Save character n-gram vectors
    f_ngrams = open('vector/%s' % eventlog + '_ngram_vectors_FastText_noTime_noEnd_Vector_vLoss_v1' + '.txt', 'w')
    f_ngrams.write('{} {}\n'.format(len(char_ngrams_dict), dim))

    for ngram, ngram_id in char_ngrams_dict.items():
        ngram_vec = word_vectors[ngram_id]
        str_vec = ' '.join(map(str, list(ngram_vec)))
        f_ngrams.write('{} {}\n'.format(ngram, str_vec))

    f.close()
    f_ngrams.close()

    # Save mapping of words to their n-grams
    f_word_ngrams = open('vector/%s' % eventlog + '_word_ngrams_FastText_noTime_noEnd_Vector_vLoss_v1' + '.txt', 'w')
    for word_id, ngrams in word_to_ngrams.items():
        if word_id in rev_vocabulary:
            word = rev_vocabulary[word_id]
            ngrams_str = ' '.join(map(str, ngrams))
            f_word_ngrams.write('{} {} {}\n'.format(word_id, word, ngrams_str))

    f_word_ngrams.close()