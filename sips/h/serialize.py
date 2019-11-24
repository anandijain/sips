import pandas as pd
import numpy as np

import tensorflow as tf

from sips.macros import bov as bm
from sips.h import hot
import sips.h.helpers as h


def serialize_row(row, hot_maps=None, to_numpy=False, include_teams=False):
    """

    """
    if isinstance(row, pd.core.series.Series):
        pass
    elif isinstance(row, list):
        row = pd.DataFrame(row, columns=bm.LINE_COLUMNS)

    if not hot_maps:
        hot_maps = hot.all_hot_maps(output="dict")

    ret = serialize_df(row, hot_maps=hot_maps)

    if to_numpy:
        ret = np.array(ret, dtype=np.float32)

    return ret


def serialize_dfs(
    dfs,
    in_cols,
    label_cols,
    replace_dict=None,
    hot_maps=None,
    to_numpy=True,
    norm=True,
    dropna=True,
    dont_hot=False,
    drop_labs=True,
    drop_extra_cols=["a_ou", "h_ou"],
):
    """
    label_cols is a subset of incols
    """
    sXs = []
    sYs = []
    if dont_hot:
        hot_maps = None
    else:
        if not hot_maps:
            hot_maps = hot.all_hot_maps(output="dict")

    if not replace_dict:
        replace_dict = {"None": np.nan, "EVEN": 100}  # hacky

    for df in dfs:
        if df is None:
            continue

        sdf = serialize_df(
            df, replace_dict=replace_dict, hot_maps=hot_maps, dropna=dropna
        )
        # typed_df = sdf.astype(np.float32)

        y = sdf[label_cols].copy()
        X = sdf.copy()

        if drop_labs:
            X = sdf.drop(label_cols, axis=1)
            try:
                X = X.drop(drop_extra_cols, axis=1)
            except KeyError:
                pass

        if to_numpy:
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            if norm:
                X = tf.keras.utils.normalize(X)
        sXs.append(X)
        sYs.append(y)

    return sXs, sYs


def serialize_df(df, replace_dict=None, hot_maps=None, dropna=True):
    """
    built to take in dataframe from running lines.py
    df: pd.DataFrame
    replace_dict: dict replaces values like 'None', and 'E
    """

    if not replace_dict:
        replace_dict = {"None": np.nan, "EVEN": 100}  # hacky

    df.replace(replace_dict, inplace=True)

    if hot_maps:
        df = hot.hot(df, hot_maps=hot_maps)

    if dropna:
        df.dropna(inplace=True)

    ret = df
    return ret


def test_sdfs():

    dfs = h.get_dfs()
    cols = bm.TO_SERIALIZE
    maps = hot.all_hot_maps()

    numbers = serialize_dfs(
        dfs, in_cols=cols, label_cols=["a_ml", "h_ml"], hot_maps=maps
    )

    print(numbers)
    return numbers


if __name__ == "__main__":
    test_sdfs()
    # pd.read_csv('')
