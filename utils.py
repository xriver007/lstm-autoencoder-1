import os
import random
from collections import Counter
import numpy as np
from data import filename_by_index


def titlize(string, line='=', margin_top=True, margin_bottom=False):
    length = len(string)

    if margin_top:
        print()
    if line is not None:
        print(line * length)

    print(string)

    if line is not None:
        print(line * length)
    if margin_bottom:
        print()

def load_data(volumes, truncate=None):
    data = []
    for i in volumes:
        with open(filename_by_index('./data', i), encoding='utf-8') as f:
            text = f.read()
            text = text if not truncate else text[:truncate]
            data.append(text)
    return '\n'.join(data)


def create_chars(data, p=0.8):
    counter = Counter(data)
    chars =  {c for c, n in counter.most_common(int(len(counter) * p))}
    chars.add(None)
    return chars


def create_mappings(chars):
    chars_to_indices = {c : i for i, c in enumerate(chars)}
    indices_to_chars = {i: c for c, i in chars_to_indices.items()}
    return chars_to_indices, indices_to_chars


def c2i(c, chars_to_indices):
    i = chars_to_indices.get(c)
    if i is None:
        return chars_to_indices[None]
    else:
        return i


def i2c(i, indices_to_chars):
    return indices_to_chars[i]


def encode(string, chars_to_indices):
    return np.array([c2i(c, chars_to_indices) for c in string])


def encode_hot(char, chars_to_indices):
    char_size = len(chars_to_indices)
    v = np.zeros(char_size)
    v[c2i(char, chars_to_indices)] = 1
    return v


def encode_hot_string(string, chars_to_indices):
    char_size = len(chars_to_indices)
    v = np.zeros((len(string), char_size))
    for i, c in enumerate(string):
        v[i, c2i(c, chars_to_indices)] = 1
    return v


def decode(indices, indices_to_chars):
    s = [i2c(i, indices_to_chars) for i in indices]
    return ''.join(s)


def decode_hot(vector, indices_to_chars):
    index = np.argmax(vector)
    return indices_to_chars[idex]


def sample(prediction, temperature=1.0):
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / temperature
    exps = np.exp(prediction)
    probs = exps / np.sum(exps)
    sample = np.random.multinomial(1, probs)
    return np.argmax(sample)


def training_set_generator(data, chars_to_indices, indices_to_chars, 
                           time, step, batch_size):
    data_size = len(data)
    char_size = len(chars_to_indices)
    maxindex = len(data) - time - 1
    i = 0
    while True:
        input_sentences = []
        output_sentences = []
        for _ in range(batch_size):
            input_sentences.append(data[i:i+time])
            output_sentences.append(data[i+1:i+1+time])
            i += step

            # reset the cursor at the every end of the cycle 
            if i >= maxindex:
                i = 0

        X = np.array([encode(s, chars_to_indices) for s in input_sentences])
        X = X.reshape(-1, time)
        y = np.zeros((X.shape[0], X.shape[1], char_size))
        for k, s in enumerate(output_sentences):
            y[k, :, :] = encode_hot_string(s, chars_to_indices)

        yield X, y


def train_on_volume(model, chars_to_indices, indices_to_chars, 
                    time, step, volume, batch_size=32, epoch=2):
    titlize('Start training on data volume {}'.format(volume))
    data = load_data([volume])
    total_batch = len(list(range(0, len(data) - time - 1, step)))
    generator = training_set_generator(
            data, chars_to_indices, indices_to_chars, 
            time, step, batch_size,
    )
    model.fit_generator(generator, total_batch, nb_epoch=epoch)


def generate(model, seed, time, chars_to_indices, indices_to_chars,
             length=400, temperature=1.0):
    generated = [*seed]
    sentence = [*seed]
    for i in range(int(length / time)):
        x = np.zeros((1, time))
        x[0] = encode(sentence, chars_to_indices)
        predictions = model.predict(x)[0]
        for prediction in predictions:
            sampled_index = sample(prediction)
            sampled_char = indices_to_chars[sampled_index]
            generated.append('<UNKNOWN>' if sampled_char is None else sampled_char) 
            sentence = sentence[1:] + [sampled_char]
    return ''.join(generated)
