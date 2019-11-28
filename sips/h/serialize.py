import pandas as pd
import numpy as np

import tensorflow as tf

from sips.macros import bov as bm
from sips.h import hot
from sips.h import helpers as h


def serialize_dfs(
    dfs,
    in_cols=None,
    label_cols=None,
    replace_dict=None,
    hot_maps=None,
    to_numpy=True,
    norm=True,
    astype=None,
    dropna=True,
    dont_hot=False,
    drop_labels=True,
    drop_extra_cols=["a_ou", "h_ou"],
    drop_cold=True,
    verbose=False,
):
    """
    Multipurpose conversion of multi-datatype DataFrames into numbers.

    Args:
        Required:
            dfs (list pd.DataFrame): DataFrame to serialize (required)
        Optional:
            in_cols (list str): training data
            label_cols (list str): label data
            replace_dict (dict): replace values in DataFrame
            hot_maps (list dict): provide your own hot maps
            to_numpy (bool): convert the dfs to np arrays 
            norm (bool): normalize using tf.keras.utils.normalize
            astype (numeric type): given a type to convert dfs to
            dropna (bool): pd.dropna 
            dont_hot (bool)
            drop_labels (bool): drop labels from training data
            drop_extra_cols (list str)
            drop_cold (bool): drop the categorical columns
            verbose (bool): print
    Returns: 
        pd.DataFrame or np.array

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
            astype=astype,
            dropna=dropna,
            drop_extra_cols=drop_extra_cols,
            drop_labels=drop_labels,
            drop_cold=drop_cold,
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
        print(f"serialized_dfs: {ret}")

    return ret


def serialize_df(
    df,
    in_cols=None,
    label_cols=None,
    replace_dict=None,
    hot_maps=None,
    to_numpy=True,
    norm=True,
    astype=None,
    dropna=True,
    dont_hot=False,
    drop_labels=True,
    drop_extra_cols=["a_ou", "h_ou"],
    drop_cold=True,
    verbose=False,
):
    """
    Multipurpose conversion of multi-datatype DataFrame into numbers.

    Args:
        Required:
            dfs (list pd.DataFrame): DataFrame to serialize (required)
        Optional:
            in_cols (list str): training data
            label_cols (list str): label data
            replace_dict (dict): replace values in DataFrame
            hot_maps (list dict): provide your own hot maps
            to_numpy (bool): convert the dfs to np arrays 
            norm (bool): normalize using tf.keras.utils.normalize
            astype (numeric type): given a type to convert dfs to
            dropna (bool): pd.dropna 
            dont_hot (bool)
            drop_labels (bool): drop labels from training data
            drop_extra_cols (list str)
            drop_cold (bool): drop the categorical columns
            verbose (bool): print
    Returns: 
        pd.DataFrame or np.array

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

        df = hot.hot(df, hot_maps=hot_maps, drop_cold=drop_cold)

    if dropna:
        df.dropna(inplace=True)

    X = df.copy()

    if label_cols is not None:
        y = df[label_cols].copy()

        if drop_labels:
            X = df.drop(label_cols, axis=1)

    if to_numpy:
        X = np.array(X, dtype=np.float32)
        if label_cols is not None:
            y = np.array(y, dtype=np.float32)
        if norm:
            X = tf.keras.utils.normalize(X)
    elif astype:
        X = X.astype(astype, errors="ignore")

        if label_cols is not None:
            y = y.astype(astype, errors="ignore")

        if norm and not isinstance(X, np.object):
            X = tf.keras.utils.normalize(X)

    if label_cols is not None:
        ret = (X, y)
    else:
        ret = X

    if verbose:
        print(f"serialized_df: {ret}")
    return ret


if __name__ == "__main__":
    pass
