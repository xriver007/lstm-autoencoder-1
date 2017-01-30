import os
import random
from collections import Counter
import numpy as np


def _c2i(c, chars_to_indices):
    i = chars_to_indices.get(c)
    if i is None:
        return chars_to_indices[None]
    else:
        return i


def _i2c(i, indices_to_chars):
    return indices_to_chars[i]


def _encode(string, chars_to_indices):
    return np.array([_c2i(c, chars_to_indices) for c in string])


def _encode_hot(char, chars_to_indices):
    char_size = len(chars_to_indices)
    v = np.zeros(char_size)
    v[_c2i(char, chars_to_indices)] = 1
    return v


def _encode_hot_string(string, chars_to_indices):
    char_size = len(chars_to_indices)
    v = np.zeros((len(string), char_size))
    for i, c in enumerate(string):
        v[i, _c2i(c, chars_to_indices)] = 1
    return v


def _decode(indices, indices_to_chars):
    s = [_i2c(i, indices_to_chars) for i in indices]
    return ''.join(s)


def _decode_hot(vector, indices_to_chars):
    index = np.argmax(vector)
    return indices_to_chars[idex]


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


def create_chars(data, p=0.8):
    counter = Counter(data)
    chars =  {c for c, n in counter.most_common(int(len(counter) * p))}
    chars.add(None)
    return chars


def create_mappings(chars):
    chars_to_indices = {c : i for i, c in enumerate(chars)}
    indices_to_chars = {i: c for c, i in chars_to_indices.items()}
    return chars_to_indices, indices_to_chars
