import os.path
import time as timemodule
import logging
import json
import random
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import (
    LSTM, 
    Bidirectional,
    Embedding, 
    Dense,
    TimeDistributed, 
    BatchNormalization, 
    Activation,
    Dropout, 
)
import numpy as np
from utils import (
    load_data, 
    create_chars,
    create_mappings,
    encode, 
    decode, 
    encode_hot,
    decode_hot,
    generate,
    train_on_volume,
)


# =================
# character mapping
# =================

data = load_data([1, 2, 3, 4, 5])
chars = create_chars(data, 0.95)
char_size = len(chars)
chars_to_indices, indices_to_chars = create_mappings(chars)
del data
del chars
print('=' * 35)
print('character map for {} chars loaded'.format(char_size))

# ===============
# hyperparameters
# ===============

time = 140
step = 70

lr = 0.03
word2vec = 256
lstm_hiddens = [256, 256]
lstm_dense = True
lstm_bidirectional = True
dense_hiddens = [256]
dropout = 0.5

# ====================
# show hyperparameters
# ====================

print('=' * 35)
print('\n'.join([
    'time span: {}',
    'step size: {}', 
    'learning rate: {}', 
    'word2vec size: {}',
    'lstm hidden size: {}',
    'dense hidden size: {}',
    'dropout rate: {}',
]).format(
    time, step, lr, word2vec, 
    lstm_hiddens, dense_hiddens, dropout, 
))

# ==============
# configurations
# ==============

lstm_name = 'lstm'
lstm_name += '-dense' if lstm_dense else ''
lstm_name += '-bidir' if lstm_bidirectional else ''
model_name = (
    'embedding{} - {}{} - {}dense ' + 
    '| timespan {} | dropout {} | charsize {}'
).format(
    word2vec, lstm_hiddens, lstm_name,
    dense_hiddens, time, dropout, char_size
)
weight_dir = './weights'
weight_path = os.path.join(weight_dir, '{}.hd5'.format(model_name))
train_history = []
volumes_per_checkpoint = 1

# ================================
# model definition and compilation
# ================================

model = Sequential()
model.add(Embedding(char_size, word2vec, input_length=time))

for h in lstm_hiddens:
    model.add(Bidirectional(LSTM(
        h, input_length=time, return_sequences=True
    )))
    if lstm_dense:
        model.add(TimeDistributed(Dense(h)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

for h in dense_hiddens:
    model.add(TimeDistributed(Dense(h)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

model.add(TimeDistributed(Dense(char_size, activation='softmax')))
model.compile(
    optimizer=RMSprop(lr=lr), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
model.summary()

# ===================
# train over datasets
# ===================

def train(data_volumes, batch=32, epoch=1):
    for volume in data_volumes:
        train_history.append(volume)
        train_on_volume(
            model, chars_to_indices, indices_to_chars, 
            time, step, volume, batch_size=batch, epoch=epoch
        )
        if len(train_history) % volumes_per_checkpoint == 0:
            if not os.path.exists(weight_dir):
                os.mkdir(weight_dir)
            model.save_weights(weight_path)


def generate_text(volume=1, temperature=1.0):
    data = load_data([volume])
    data_size = len(data)
    seed_index = random.randint(0, data_size - 1 - time)
    seed = data[seed_index:seed_index+time]
    encoded_seed = encode(seed, chars_to_indices)
    print('=' * 40)
    print('text generation seed: {}'.format(seed))

    generated = generate(model, seed, time, chars_to_indices, indices_to_chars, 
                         temperature=temperature)
    print('=' * 40)
    print('generated text: {}'.format(generated))

