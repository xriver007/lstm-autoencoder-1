import numpy as np
from data import load_data
from utils import (
    titlize,
    _encode, 
    _encode_hot_string
)


def training_set_generator(data, c2i, time_size, step_size, batch_size):
    data_size = len(data)
    char_size = len(c2i)
    maxindex = len(data) - time_size - 1
    i = 0
    while True:
        input_sentences = []
        output_sentences = []
        for _ in range(batch_size):
            input_sentences.append(data[i:i+time_size])
            output_sentences.append(data[i:i+time_size])
            i += step_size

            # reset the cursor at the every end of the cycle 
            if i >= maxindex:
                i = 0

        X = np.array([_encode(s, c2i) for s in input_sentences])
        X = X.reshape(-1, time_size)
        y = np.zeros((X.shape[0], X.shape[1], char_size))
        for k, s in enumerate(output_sentences):
            y[k, :, :] = _encode_hot_string(s, c2i)

        yield X, y


def train(model, volumes, c2i, 
          time_size, step_size,
          batch_size=32, epoch=5):
    titlize('Start Training on Data Volume {}'.format(volumes))
    data = load_data(volumes)
    total_batch = len(list(range(0, len(data) - time_size - 1, step_size)))
    generator = training_set_generator(
        data, c2i,
        time_size=time_size,
        step_size=step_size, 
        batch_size=batch_size,
    )
    model.fit_generator(generator, total_batch, nb_epoch=epoch)
