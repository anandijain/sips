import pandas as pd
import numpy as np

import tensorflow as tf

from sips.macros import bov as bm
from sips.h import hot
from sips.h import helpers as h


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
    in_cols=None,
    label_cols=None,
    replace_dict=None,
    hot_maps=None,
    to_numpy=True,
    norm=True,
    dropna=True,
    dont_hot=False,
    drop_labs=True,
    drop_extra_cols=["a_ou", "h_ou"],
    verbose=False
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

    if isinstance(dfs, dict):  # hacky
        dfs = list(dfs.values())

    for df in dfs:
        sdf = serialize_df(
            df,
            in_cols=in_cols,
            label_cols=label_cols,
            replace_dict=replace_dict,
            hot_maps=hot_maps,
            to_numpy=to_numpy,
            norm=norm,
            dropna=dropna,
            drop_extra_cols=drop_extra_cols,
            drop_labs=drop_labs,
        )
        if label_cols is not None:
            X, y = sdf
            sYs.append(y)
        else:
            X = sdf

        sXs.append(X)

    if label_cols is not None:
        ret = (sXs, sYs)
    else:
        ret = sXs

    if verbose:
        print(f'serialized_dfs: {ret}')
        
    return ret


def serialize_df(
    df,
    in_cols=None,
    label_cols=None,
    replace_dict=None,
    hot_maps=None,
    to_numpy=True,
    norm=True,
    dropna=True,
    dont_hot=False,
    drop_labs=True,
    drop_extra_cols=["a_ou", "h_ou"],
    verbose=False
):
    """
    multipurpose conversion of multi-datatype DataFrames into numbers

    Args:
        df (pd.DataFrame) -- DataFrame to serialize (required)
        in_cols (list df.column 
        replace_dict : dict -- replace values in DataFrame
    
    --
    returns: pd.DataFrame or np array

    norm is only applied if to_numpy is True
    """
    if drop_extra_cols is not None:
        try:
            df.drop(drop_extra_cols, axis=1, inplace=True)
        except KeyError:
            pass

    if not replace_dict:
        replace_dict = {"None": np.nan, "EVEN": 100}

    df.replace(replace_dict, inplace=True)

    if dont_hot:
        hot_maps = None
    else:
        if not hot_maps:
            hot_maps = hot.all_hot_maps(output="dict")

        df = hot.hot(df, hot_maps=hot_maps)

    if dropna:
        df.dropna(inplace=True)

    X = df.copy()

    if label_cols is not None:
        y = df[label_cols].copy()

        if drop_labs:
            X = df.drop(label_cols, axis=1)

    if to_numpy:
        X = np.array(X, dtype=np.float32)
        if label_cols is not None:
            y = np.array(y, dtype=np.float32)

        if norm:
            X = tf.keras.utils.normalize(X)

    if label_cols is not None:
        ret = (X, y)
    else:
        ret = X

    if verbose:
        print(f'serialized_df: {ret}')
    return ret


def test_sdfs():
    dfs = h.get_dfs()
    cols = bm.TO_SERIALIZE
    maps = hot.all_hot_maps()

    numbers = serialize_dfs(
        dfs, in_cols=None, label_cols=None, hot_maps=maps, to_numpy=False
    )

    print(numbers)
    return numbers


if __name__ == "__main__":
    test_sdfs()
