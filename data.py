from operator import itemgetter
import os
import random


_DATA_DELIMITER = '+'


def shuffle_and_index(data_dir='./data'):
    files = os.listdir(data_dir)
    random.shuffle(files)
    for i, f in enumerate(files):
        old = os.path.join(data_dir, f)
        new = os.path.join(data_dir, '{}{}{}'.format(
            i, _DATA_DELIMITER, f.split(_DATA_DELIMITER)[1] if 
            _DATA_DELIMITER in f else f
        ))
        os.rename(old, new)
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir)]


def data_by_index(data_dir, index):
    if index is None:
        return None

    files = os.listdir(data_dir)
    assert all([
        f.split(_DATA_DELIMITER)[0].isdigit() for f in files
    ]), 'data not indexed'
    files = sorted(files, key=lambda f: int(f.split(_DATA_DELIMITER)[0]))

    if isinstance(index, int):
        found = files[index]
        return os.path.join(data_dir, found)
    else:
        if not len(index):
            return []
        else:
            found = itemgetter(index)(files)
            return [os.path.join(data_dir, f) for f in found]


def load_data(volumes, truncate=None):
    data = []
    for i in volumes:
        with open(data_by_index('./data', i), encoding='utf-8') as f:
            text = f.read()
            text = text if not truncate else text[:truncate]
            data.append(text)
    return '\n'.join(data)
