import math

import pandas as pd
import numpy as np


def series_mean(series):
    items = series.tolist()
    tot = 0

    for item in items:
        tot += item

    mean = tot / len(items)
    return mean


def series_std_dev(series, mean):
    items = series.tolist()
    var = 0

    for item in items:
        var += (item - mean) ** 2

    std_dev = math.sqrt(var / len(items))
    return std_dev


def df_means(df):
    list_of_mean = []

    for col in df:
        mean = series_mean(df[col])
        list_of_mean.append(mean)

    return list_of_mean


def df_std_devs(df, df_means):
    list_of_std_dev = []

    for index, col in enumerate(df):
        list_of_std_dev.append(series_std_dev(df[col], df_means[index]))

    return list_of_std_dev


def df_normalize(df):
    means = df_means(df)
    devs = df_std_devs(df, means)
    normed_list = []

    for index, col in enumerate(df):
        for elt in df[col]:
            try:
                normed_list.append((elt - means[index] / devs[index]))
            except ZeroDivisionError:
                continue

    return normed_list

