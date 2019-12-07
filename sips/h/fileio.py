"""
helper functions for writing to csvs

"""
import os
import pandas as pd

from sips.macros import tfm


def init_csv(file_name: str, header: list, close=True):
    """
    create a csv and write header, option to close

    """
    csv_file = open(file_name, "a")
    write_list(csv_file, header)
    if close:
        csv_file.close()
    else:
        return csv_file


def write_list(file, list):
    """
    write a list to a csv

    """
    if not list:
        return
    length = len(list)
    for i, elt in enumerate(list):
        file.write(str(elt))
        if i == length - 1:
            file.write("\n")
        else:
            file.write(",")


def append_csv(fn: str, df: pd.DataFrame):
    """

    """
    with open(fn, "a") as f:
        df.to_csv(f, header=False)


def get_fns_generator(directory):
    """
    return the absolute file paths of a directory

    """
    for dirpath, _, filenames in os.walk(directory):
        for file_name in filenames:
            yield os.path.abspath(os.path.join(dirpath, file_name))


def get_fns(directory):
    """
    return the absolute file paths of a directory

    """
    fns_generator = get_fns_generator(directory)
    return list(fns_generator)
