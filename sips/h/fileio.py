import os
import random


def init_csv(fn, header, close=True):
    """

    """
    f = open(fn, "a")
    write_list(f, header)
    if close:
        f.close()
    else:
        return f


def write_list(file, list):
    if not list:
        return
    length = len(list)
    for i, elt in enumerate(list):
        file.write(str(elt))
        if i == length - 1:
            file.write("\n")
        else:
            file.write(",")


def get_fns(dir):
    fns = os.listdir(dir)
    try:
        fns.remove("LOG.csv")
    except ValueError:
        pass

    return fns


def train_test_split_dir(fns, train_frac=0.7, shuffle=False):
    """

    """
    num_files = len(fns)
    split_idx = round(0.7 * num_files)

    if shuffle:
        random.shuffle(fns)

    train_fns = fns[0:split_idx]
    test_fns = fns[split_idx:]

    return train_fns, test_fns
