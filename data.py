from operator import itemgetter
import os
import random


def shuffle_and_index(data_dir):
    files = os.listdir(data_dir)
    random.shuffle(files)
    for i, f in enumerate(files):
        old = os.path.join(data_dir, f)
        new = os.path.join(data_dir, '{}-{}'.format(i, f))
        os.rename(old, new)
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir)]


def filename_by_index(data_dir, index):
    if index is None:
        return None

    files = os.listdir(data_dir)
    assert all([f.split('-')[0].isdigit() for f in files]), 'data not indexed'
    files = sorted(files, key=lambda f: int(f.split('-')[0]))

    if isinstance(index, int):
        found = files[index]
        return os.path.join(data_dir, found)
    else:
        if not len(index):
            return []
        else:
            found = itemgetter(index)(files)
            return [os.path.join(data_dir, f) for f in found]
