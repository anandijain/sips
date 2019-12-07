"""

"""
import pandas as pd
import numpy as np

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
    norm=False,
    astype=None,
    dropna=True,
    filter_empty=True,
    dont_hot=False,
    drop_labels=True,
    drop_extra_cols=["a_ou", "h_ou"],
    drop_cold=True,
    output_type="list",
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
    if output_type == "dict":
        ret = {}

    serialized_Xs = []
    serialized_Ys = []

    if dont_hot:
        hot_maps = None
    else:
        if not hot_maps:
            hot_maps = hot.all_hot_maps(output="dict")

    if isinstance(dfs, dict):  # hacky
        dfs = list(dfs.values())
    elif isinstance(dfs, str):
        dfs = h.get_dfs()

    for df in dfs:
        serialized_df = serialize_df(
            df,
            in_cols=in_cols,
            label_cols=label_cols,
            replace_dict=replace_dict,
            hot_maps=hot_maps,
            to_numpy=to_numpy,
            norm=norm,
            astype=astype,
            dropna=dropna,
            dont_hot=dont_hot,
            drop_extra_cols=drop_extra_cols,
            drop_labels=drop_labels,
            drop_cold=drop_cold,
            output_type=output_type,
        )

        if filter_empty:
            if serialized_df is None:
                continue

        if output_type == "dict":
            game_id, data = serialized_df
            ret[game_id] = data
        elif label_cols is not None:
            features, labels = serialized_df
            serialized_Ys.append(labels)
            serialized_Xs.append(features)
        else:
            features = serialized_df
            serialized_Xs.append(features)

    if output_type == "dict":
        return ret

    if label_cols is not None:
        ret = (serialized_Xs, serialized_Ys)
    else:
        ret = serialized_Xs

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
    norm=False,
    astype=None,
    dropna=True,
    dont_hot=False,
    drop_labels=True,
    drop_extra_cols=["a_ou", "h_ou"],
    drop_cold=True,
    output_type="list",  # list or dict
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
            drop_extra_cols (list str): disjoint with union(in_cols, label)
            drop_cold (bool): drop the categorical columns
            verbose (bool): print
    Returns: 
        pd.DataFrame or np.array

    """
    if isinstance(df, str):
        print(f"dataframe: {df} is of type string for some reason")

    if drop_extra_cols is not None:
        try:
            df.drop(drop_extra_cols, axis=1, inplace=True)
        except KeyError:
            pass

    if in_cols and label_cols:
        print(f"in_cols: {in_cols}")
        print(f"label_cols: {label_cols}")
        all_cols = list(set(in_cols + label_cols))
        df = df[all_cols].copy()
    elif in_cols and not label_cols:
        df = df[in_cols].copy()

    if not replace_dict:
        replace_dict = {"None": np.nan, "EVEN": 100}

    try:
        df.replace(replace_dict, inplace=True)
    except TypeError:
        pass

    if dont_hot:
        hot_maps = None
    else:
        if not hot_maps:
            hot_maps = hot.all_hot_maps(output="dict")

        df = hot.hot(df, hot_maps=hot_maps, drop_cold=drop_cold)

    if dropna:
        df.dropna(inplace=True)

    if df.empty:
        return

    game_id = df.game_id.iloc[0]
    features = df.copy()

    if label_cols is not None:
        labels = df[label_cols].copy()

        if drop_labels:
            features = df.drop(label_cols, axis=1)

    if to_numpy:
        features = np.array(features, dtype=np.float32)
        if label_cols is not None:
            labels = np.array(labels, dtype=np.float32)
        if norm:
            features = h.sk_scale(features, to_df=False)

    elif astype:
        features = features.astype(astype, errors="ignore")

        if label_cols is not None:
            labels = labels.astype(astype, errors="ignore")

        if norm and not isinstance(features, np.object):
            features = h.sk_scale(features, to_df=True)

    if label_cols is not None:
        ret = (features, labels)
    else:
        ret = features

    if output_type == "dict":
        return (game_id, ret)

    if verbose:
        print(f"serialized_df: {ret}")
    return ret


if __name__ == "__main__":
    cols = ["last_mod", "quarter", "secs", "a_pts", "h_pts", "a_ml", "h_ml"]
    dfs = h.get_dfs()
    df = dfs[0]
    print(type(df))
    data = serialize_dfs(dfs, in_cols=cols, norm=True, dont_hot=True)
    print(data[0])
