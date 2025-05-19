# -*- coding = UTF-8 -*-
# @Time : 2025/3/16 16:26
# @Author : 付传萌
# @File : trainVector_ELMO.py
# @Software : PyCharm

import numpy as np

np.random.seed(13)
import csv
import time
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, Lambda, Input, Bidirectional, LSTM, Dropout, Concatenate
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from datetime import datetime
import matplotlib.pyplot as plt

# Configuration
eventlog = 'Codeforces 936_15'
dim = 3  # Increased dimension for ELMo
window_size = 1
lstm_units = 32  # LSTM units for ELMo model


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
    f = open('vector/%s' % eventlog + '_ELMo_noTime_noEnd_vocabulary' + '.txt', 'w', encoding='utf-8')
    for k in vocabulary:
        f.write(str(k) + '\t' + str(vocabulary[k]) + '\n')
    return vocabulary


vocabulary = makeVocabulary(data, eventlog)


# Data preprocessing (same as original)
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


# Generate training data sequences
def generate_elmo_data(corpus, window_size, V):
    """Generate training data for ELMo model"""
    for words in corpus:
        L = len(words)
        if L < 2 * window_size + 1:  # Skip sequences that are too short
            continue

        for index in range(window_size, L - window_size):
            # Get context for forward LSTM
            forward_context = [words[i][1] for i in range(index - window_size, index)]
            # Get context for backward LSTM
            backward_context = [words[i][1] for i in range(index + 1, index + window_size + 1)]
            # Get target
            target = words[index][1]

            # Yield both contexts and target
            yield (forward_context, backward_context, target)


# Prepare training data
X_forward = []
X_backward = []
Y = []

for forward_context, backward_context, target in generate_elmo_data(data_merge, window_size, vocabulary_num):
    X_forward.append(forward_context)
    X_backward.append(backward_context)
    Y.append(np.eye(vocabulary_num)[target])

X_forward = np.array(X_forward)
X_backward = np.array(X_backward)
Y = np.array(Y)

if __name__ == '__main__':
    # Build ELMo model
    # Input layers
    forward_input = Input(shape=(window_size,), name='forward_input')
    backward_input = Input(shape=(window_size,), name='backward_input')

    # Embedding layers
    embedding_layer = Embedding(input_dim=vocabulary_num, output_dim=dim)
    forward_embedding = embedding_layer(forward_input)
    backward_embedding = embedding_layer(backward_input)

    # Forward LSTM (left-to-right)
    forward_lstm = LSTM(lstm_units, return_sequences=True)(forward_embedding)
    forward_lstm_2 = LSTM(lstm_units)(forward_lstm)

    # Backward LSTM (right-to-left)
    backward_lstm = LSTM(lstm_units, return_sequences=True, go_backwards=True)(backward_embedding)
    backward_lstm_2 = LSTM(lstm_units, go_backwards=True)(backward_lstm)

    # Concatenate both representations
    merged = Concatenate()([forward_lstm_2, backward_lstm_2])

    # Output layer
    output = Dense(vocabulary_num, activation='softmax')(merged)

    # Create and compile model
    elmo_model = Model(inputs=[forward_input, backward_input], outputs=output)

    # Compile model
    opt = Adam(learning_rate=0.001)
    elmo_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                   verbose=1, mode='auto', cooldown=0, min_lr=0.0001)
    model_checkpoint = ModelCheckpoint(
        'vector/%s' % eventlog + '_ELMo_noTime_noEnd_Vector_vLoss_{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_accuracy', verbose=1, save_best_only=True,
        save_weights_only=False, mode='auto')

    # Train model
    history = elmo_model.fit(
        [X_forward, X_backward], Y,
        validation_split=0.2,
        verbose=2,
        callbacks=[early_stopping, model_checkpoint, lr_reducer],
        batch_size=32,
        epochs=100
    )

    # Extract embeddings from trained model
    embeddings = embedding_layer.get_weights()[0]

    # Save embeddings
    f = open('vector/%s' % eventlog + '_vectors_ELMo_noTime_noEnd_Vector_v1' + '.txt', 'w')
    f.write('{} {}\n'.format(vocabulary_num, dim))
    for word in range(vocabulary_num):
        str_vec = ' '.join(map(str, list(embeddings[word, :])))
        f.write('{} {}\n'.format(word, str_vec))
    f.close()

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.tight_layout()
    plt.savefig('vector/%s' % eventlog + '_ELMo_training_history.png')
    plt.close()


    # Function to get contextual embeddings
    def get_elmo_embeddings(sequence, model):
        """
        Get contextual embeddings for a sequence
        """
        # Convert sequence to appropriate format
        seq_len = len(sequence)
        if seq_len < 2 * window_size + 1:
            return None

        contextual_embeddings = []

        for i in range(window_size, seq_len - window_size):
            forward_context = [sequence[j][1] for j in range(i - window_size, i)]
            backward_context = [sequence[j][1] for j in range(i + 1, i + window_size + 1)]

            # Create input arrays
            fw_input = np.array([forward_context])
            bw_input = np.array([backward_context])

            # Get intermediate layer outputs (forward and backward LSTMs)
            forward_lstm_layer = Model(inputs=model.input, outputs=model.get_layer('lstm_1').output)
            backward_lstm_layer = Model(inputs=model.input, outputs=model.get_layer('lstm_3').output)

            forward_embeddings = forward_lstm_layer.predict([fw_input, bw_input])
            backward_embeddings = backward_lstm_layer.predict([fw_input, bw_input])

            # Combine embeddings
            combined = np.concatenate([
                forward_embeddings[0, -1, :],  # Last state of forward LSTM
                backward_embeddings[0, 0, :]  # First state of backward LSTM
            ])

            contextual_embeddings.append(combined)

        return np.array(contextual_embeddings)


    # Save a sample contextual embedding for demonstration
    if len(data_merge) > 0 and len(data_merge[0]) > 2 * window_size + 1:
        sample_seq = data_merge[0]
        contextual_embeds = get_elmo_embeddings(sample_seq, elmo_model)

        if contextual_embeds is not None:
            np.save('vector/%s' % eventlog + '_sample_contextual_embeddings.npy', contextual_embeds)

            # Save a visualization of the contextual embeddings
            plt.figure(figsize=(10, 6))
            plt.imshow(contextual_embeds, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title('ELMo Contextual Embeddings Visualization')
            plt.xlabel('Dimension')
            plt.ylabel('Sequence Position')
            plt.tight_layout()
            plt.savefig('vector/%s' % eventlog + '_ELMo_contextual_embeddings_viz.png')
            plt.close()