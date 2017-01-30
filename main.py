import os.path
import numpy as np
from data import load_data
from utils import (
    create_chars,
    create_mappings,
)
from train import train
from autoencoder import Autoencoder


# ===============
# hyperparameters
# ===============

volumes_for_chars = 100
volumes_for_training = 100
batch_size = 32
epoch = 5
time_size = 140
step_size = 140
lr = 0.03
word2vec_size = 512
lstm_hiddens = [256, 128, 64, 32]
lstm_dense = True
lstm_bidirectional = True
dropout_rate = 0.5

# =================
# character mapping
# =================

data = load_data(range(volumes_for_chars))
chars = create_chars(data, 0.95)
char_size = len(chars)
c2i, i2c = create_mappings(chars)
del data
del chars
print('=' * 35)
print('character map for {} chars loaded'.format(char_size))

# ====================
# show hyperparameters
# ====================

print('=' * 35)
print('\n'.join([
    'time size: {}',
    'step size: {}', 
    'batch_size: {}',
    'learning rate: {}', 
    'word2vec size: {}',
    'lstm hidden sizes: {}',
    'lstm dense: {}',
    'dropout rate: {}',
]).format(
    time_size, 
    step_size, 
    batch_size,
    lr, 
    word2vec_size, 
    lstm_hiddens, 
    lstm_dense,
    dropout_rate, 
))

config = {
    'lr': lr,
    'step_size': step_size,
    'time_size': time_size,
    'word2vec_size': word2vec_size,
    'lstm_hiddens': lstm_hiddens,
    'lstm_dense': lstm_dense,
    'dropout_rate': dropout_rate,
}


# ==============
# train the model
# ==============

autoencoder = Autoencoder(dimension=char_size, config=config)
train(
    autoencoder.model, range(volumes_for_training), c2i, 
    time_size=time_size,
    step_size=step_size,
    batch_size=batch_size,
    epoch=epoch
)
