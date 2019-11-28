import io
import os

from sips.macros import tfm

def init_csv(fn, header, close=True):
    # create a csv and write header, option to close
    f = open(fn, "a")
    write_list(f, header)
    if close:
        f.close()
    else:
        return f


def write_list(file, list):
    # write a list to a csv
    if not list:
        return
    length = len(list)
    for i, elt in enumerate(list):
        file.write(str(elt))
        if i == length - 1:
            file.write("\n")
        else:
            file.write(",")


def absolute_file_paths(directory):
    # return the absolute file paths of a directory
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def get_fns(directory=tfm.READ_FROM):
    # read a directiory and return absolute paths
    fns = list(absolute_file_paths(directory))
    try:
        fns.remove(directory + "LOG.csv")
    except ValueError:
        pass
    return fns
